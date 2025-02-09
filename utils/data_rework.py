"""
Note that some code in this script is adapted or modified from PepLand and PharmHGT.
This dataloader will only read one SDF file.
Discarded the split function.
"""

import json
import os

import torch
import dgl

from .FraSCESS.mol2heterograph_FraSCESS import mol2heterograph_frascess
from .RegularMethod.BRICSFrag import mol2heterograph_BRICS

from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
from utils.std_logger import info
from tqdm import tqdm

class MoleculeDataLoader(Dataset):
    def __init__(self, cfg, stage, mask_graph, sdf_path):
        self.cfg = cfg
        self.stage = stage
        self.mask_graph = mask_graph
        self.vocab_dict = None
        if self.cfg.logger.test_run:
            info("Test run enabled.")
        # Load vocabulary
        self.load_vocabdict(cfg.data.vocablist)

        # Pre-process the entire SDF file into (graph, label) pairs
        self.graphs_labels = self.mol_to_heterograph(sdf_path)

    def mol_to_heterograph(self, sdf_path):
        """Reads all molecules from sdf_path, converts each to a DGL heterograph,
           and returns a list of (graph, label) tuples.
        """
        # 1) Read all molecules
        mols = read_SMI_file(sdf_path)
        
        # 2) Convert each RDKit mol to a DGL graph
        graphs_labels = []
        for mol in tqdm(mols, desc=f"Processing {os.path.basename(sdf_path)}", unit="mol"):
            if self.cfg.train.frag_method == "FraSCESS":
                g = mol2heterograph_frascess(mol, mol.GetProp("_Name"), self.vocab_dict)
            elif self.cfg.train.frag_method == "AdaFrag":
                from .Pepland.data import Mol2HeteroGraph as Pepland_Mol2HG
                g = Pepland_Mol2HG(mol)
            elif self.cfg.train.frag_method == "BRICS":
                g = mol2heterograph_BRICS(mol, mol.GetProp("_Name"), self.vocab_dict)
            else:
                raise ValueError(f"Unknown fragmentation method: {self.cfg.train.frag_method}")
            
            if g.number_of_nodes() == 0:
                raise ValueError("Empty graph created.")
            
            # 3) Determine label based on the file name or some SDF property
            if "unlabel" in sdf_path:
                label = -1
            elif "positive" in sdf_path:
                label = 1
            elif "negative" in sdf_path:
                label = 0
            elif self.cfg.logger.test_run:
                label = -1
            else:
                raise ValueError("Invalid SDF file name or labeling logic not handled.")
            
            if self.mask_graph and self.cfg.train.train_tag == "s1_pretrain":
                g = self.mask_graph(g)

            graphs_labels.append((g, label))
        
        return graphs_labels

    def load_vocabdict(self, vocabjson_path):
        with open(vocabjson_path, 'r') as f:
            data = json.load(f)
        
        frequent_fragments = data.get("frequent_fragments", {})
        self.vocab_dict = {
            fragment: i
            for i, (fragment, count) in enumerate(frequent_fragments.items())
        }

        info(f"Loaded {len(self.vocab_dict)} fragments from {vocabjson_path}")

        # info("Vocab dict:", vocab_dict) {0: 'NCC=O', 1: 'CC=O', 2: 'CC(N)=O'...}

    def __getitem__(self, idx):
        return self.graphs_labels[idx]

    def __len__(self):
        return len(self.graphs_labels)

def collate_fn(batch):
    graphs, labels = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels, dtype=torch.float32)
    return batched_graph, labels

def create_dataset(cfg, stage, mask_graph):
    info(f"Creating dataset for {stage}...")
    return MoleculeDataLoader(cfg, stage, mask_graph)

def make_loaders(cfg, batch_size, mask_graph):
    # 1) Create datasets for each split
    train_dataset = MoleculeDataLoader(cfg, "train", mask_graph, cfg.data.train_sdf)
    
    if cfg.logger.test_run:
        valid_dataset = train_dataset
        test_dataset = train_dataset
    else:
        valid_dataset = MoleculeDataLoader(cfg, "valid", mask_graph, cfg.data.valid_sdf)
        test_dataset  = MoleculeDataLoader(cfg, "test", mask_graph, cfg.data.test_sdf)

    # 2) Wrap them in PyTorch DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )

    # 3) Return the loaders plus the vocabulary from one of them (or a consistent place)
    dataloaders = {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader
    }
    
    return dataloaders, train_dataset.vocab_dict

def read_SMI_file(file_path: str) -> list[Chem.Mol]:
    """
    Read multiple mols from a single SMILES file, and extract bond properties.
    
    :param file_path: Path to the SMILES file
    :return: List of RDKit Mol objects with bond properties restored
    """
    if not os.path.exists(file_path):
        raise ValueError("File path does not exist.")
    
    suppl = Chem.SmilesMolSupplier(file_path, titleLine=True, delimiter = '!^!') 
    mols = []
    
    for mol in suppl:
        if mol is None:
            continue
        if mol.HasProp("_BondProps"):
            bond_props = mol.GetProp("_BondProps").split('|')
            for prop in bond_props:
                idx, value = prop.split(':')
                bond = mol.GetBondWithIdx(int(idx))
                bond.SetProp("LinkingBond", value)
        mols.append(mol)

    if not mols:
        raise ValueError("No valid molecule found in the SMILES file")
    
    return mols

def read_SDF_file(file_path: str) -> list[Chem.Mol]:
    """
    Read multiple mols from a single sdf, and extract bond properties.
    """
    if not os.path.exists(file_path):
        raise ValueError("File path does not exsit.")
    
    suppl = Chem.SDMolSupplier(file_path)

    mols = []
    
    for mol in suppl:
        if mol is None:
            continue
        if mol.HasProp("_BondProps"):
            bond_props = mol.GetProp("_BondProps").split('|')
            for prop in bond_props:
                idx, value = prop.split(':')
                bond = mol.GetBondWithIdx(int(idx))
                bond.SetProp("LinkingBond", value)
        mols.append(mol)

    if not mols:
        raise ValueError("No valid molecule found in the SDF file")
    
    return mols