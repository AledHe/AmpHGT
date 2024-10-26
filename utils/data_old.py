"""
Note that some code in this script is adapted or modified from PepLand and PharmHGT.
"""

import json
import os
import random
from .FraSCESS.mol2heterograph_FraSCESS import mol2heterograph_frascess

from rdkit import Chem
from multiprocessing import Pool
from torch.utils.data import IterableDataset
from torch.utils.data.distributed import DistributedSampler
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm

class DataLoader(IterableDataset):
    def __init__(self, cfg, stage, mask_graph):
        self.cfg = cfg
        self.stage = stage # pretrain, finetune, or test
        self.sdf_file_paths = []
        self.vocab_dict = None
        self.mask_graph = mask_graph

    def load_SDF_directory(self, directory):
        # load all file paths in the directory
        sdf_file_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(".sdf"):
                    sdf_file_paths.append(os.path.join(root, file))
        
        if len(sdf_file_paths) < 1:
            raise ValueError(f"No SDF files found in {directory}")
        print(f"Loaded {len(sdf_file_paths)} SDF files in {directory}")
        self.sdf_file_paths.extend(sdf_file_paths)
    
    def mol_to_heterograph_worker(self, sdf_path):
        mol = read_SDF_file(sdf_path)
        g = mol2heterograph_frascess(mol, sdf_path, self.vocab_dict)
        if g.number_of_nodes() > 0:
            if self.mask_graph:
                g = self.mask_graph(g)
            return g
        return None

    def mol_to_heterograph(self):
        with Pool(processes=self.cfg.data.load_worker) as pool:
            graphs = list(tqdm(pool.imap(self.mol_to_heterograph_worker, self.sdf_file_paths), total=len(self.sdf_file_paths), desc="Converting molecules to heterographs"))
        # Filter out None results
        return [g for g in graphs if g is not None]

    def load_vocabdict(self, vocabjson_path):
        with open(vocabjson_path, 'r') as f:
            data = json.load(f)
        
        frequent_fragments = data.get("frequent_fragments", {})
        vocab_dict = {fragment: i for i, (fragment, count) in enumerate(frequent_fragments.items())}

        print(f"Loaded {len(vocab_dict)} fragments from {vocabjson_path}")

        # print("Vocab dict:", vocab_dict) {0: 'NCC=O', 1: 'CC=O', 2: 'CC(N)=O'...
        
        self.vocab_dict = vocab_dict

    def slice_path(self, path) -> list:
        # slice the path by "+".
        return path.split(" + ")
    
    def sdf_to_mol(self):
        # convert sdf to mol
        mols = []
        for sdf_path in self.sdf_file_paths:
            mol = read_SDF_file(sdf_path)
            mols.append((sdf_path, mol))
        self.mols = mols

    def __iter__(self):
        # print dir
        print(f"direcotry of dataloader: {os.getcwd()}")
        # load vocab dict
        self.load_vocabdict(self.cfg.data.vocablist)

        if self.stage == 'pretrain-s1': # unlabeled masking task
            # slice the path
            pathes = self.slice_path(self.cfg.data.unlabel_dataset_path)
            for path in pathes:
                self.load_SDF_directory(path)
        elif self.stage == 'pretrain-s2': # CAA labeled task
            pass
        elif self.stage == 'finetune': # NCAA labeled task
            pass
        else:
            raise ValueError(f"Unknown stage: {self.stage}")

        graphs = self.mol_to_heterograph()

        for g in graphs:
            yield g

    def test_loader(self):
        print("Testing DataLoader...")
        self.load_vocabdict("../tokenizer/vocablist/vocab_2.json")
        self.load_SDF_directory(self.cfg.data.unlabel_dataset_path)
        graphs = self.mol_to_heterograph()
        print(f"Number of graphs: {len(graphs)}")
        exit()

def create_dataset(cfg, stage, mask_graph):
    print(f"Creating dataset for {stage}...")
    dataset = DataLoader(cfg, stage, mask_graph)

    # dataset.test_loader()

    return dataset

def split_dataset(dataset, train_ratio=0.7, valid_ratio=0.15, seed=42):
    random.seed(seed)
    data_size = len(dataset)
    indices = list(range(data_size))
    random.shuffle(indices)

    train_split = int(train_ratio * data_size)
    valid_split = int((train_ratio + valid_ratio) * data_size)

    train_indices = indices[:train_split]
    valid_indices = indices[train_split:valid_split]
    test_indices = indices[valid_split:]

    train_dataset = [dataset[i] for i in train_indices]
    valid_dataset = [dataset[i] for i in valid_indices]
    test_dataset = [dataset[i] for i in test_indices]

    print(f"Train size {train_ratio}: {len(train_dataset)}, Valid size {valid_ratio}: {len(valid_dataset)}, Test size {valid_ratio}: {len(test_dataset)}")

    return train_dataset, valid_dataset, test_dataset

def make_loaders(cfg, stage, world_size, global_rank, batch_size, mask_graph):
    dataset = create_dataset(cfg, stage, mask_graph)

    train_dataset, valid_dataset, test_dataset = split_dataset(list(dataset), train_ratio=0.7, valid_ratio=0.15, seed=42)

    if cfg.mode.ddp:
        train_smapler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank)
        valid_smapler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=global_rank)
        test_smapler = DistributedSampler(test_dataset, num_replicas=world_size, rank=global_rank)
        train_loader = GraphDataLoader(train_dataset, batch_size=batch_size,  num_workers=1, pin_memory=True, sampler=train_smapler)
        valid_loader = GraphDataLoader(valid_dataset, batch_size=batch_size, num_workers=1, pin_memory=True, sampler=valid_smapler)
        test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, num_workers = 1, pin_memory=True, sampler=test_smapler)
    else:
        train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, num_workers=1)
        valid_loader = GraphDataLoader(valid_dataset, batch_size=batch_size, num_workers=1)
        test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, num_workers=1)

    dataloaders = {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader
    }

    return dataloaders

def read_SDF_file(file_path: str):
    """
    Read a molecule from an .sdf file and extract bond properties.
    """
    if not os.path.exists(file_path):
        raise ValueError("File path does not exist")
    
    suppl = Chem.SDMolSupplier(file_path)
    for mol in suppl:
        if mol is None:
            continue
        if mol.HasProp("_BondProps"):
            bond_props = mol.GetProp("_BondProps").split('|')
            # print(bond_props)
            for prop in bond_props:
                idx, value = prop.split(':')
                bond = mol.GetBondWithIdx(int(idx))
                bond.SetProp("LinkingBond", value)
        break  # we only want the first molecule in the file

    if mol is None:
        raise ValueError("No valid molecule found in the SDF file")
    
    return mol