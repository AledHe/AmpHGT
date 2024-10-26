"""
Note that some code in this script is adapted or modified from PepLand and PharmHGT.
"""

import hashlib
import json
import math
import os
import random
import shutil

import torch
import dgl

from .FraSCESS.mol2heterograph_FraSCESS import mol2heterograph_frascess
from .RegularMethod.BRICSFrag import mol2heterograph_BRICS

from rdkit import Chem
from torch.utils.data import IterableDataset, DataLoader
from utils.std_logger import info

copied = False

class MoleculeDataLoader(IterableDataset):
    def __init__(self, cfg, stage, mask_graph, split_ratio_t, split_ratio_v):
        self.cfg = cfg
        self.stage = stage # train, valid, test
        self.sdf_file_paths = []
        self.vocab_dict = None
        self.mask_graph = mask_graph
        self.train_paths = []
        self.valid_paths = []
        self.test_paths = []
        self.split_ratio_t = split_ratio_t
        self.split_ratio_v = split_ratio_v
        self.load_SDF_directory(cfg.data.dataset_path)
        self.load_vocabdict(cfg.data.vocablist)
        if self.cfg.train.train_tag != "eval":
            self.split_paths()
            if cfg.train.verify_splits:
                self.verify_splits()
                self.copy_files_to_split_dirs()
        else:
            self.test_paths = self.sdf_file_paths

    def load_SDF_directory(self, dirc):
        pathes = self.slice_path(dirc)
        # Load all file paths in the directory
        for directory in pathes:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(".sdf"):
                        self.sdf_file_paths.append(os.path.join(root, file))
            info(f"Loaded {len(self.sdf_file_paths)} SDF files in {directory}")
        if len(self.sdf_file_paths) < 1:
            raise ValueError(f"No SDF files found in {directory}")

    def split_paths(self):
        # Split the paths into train, valid, test
        random.seed(42)
        random.shuffle(self.sdf_file_paths)
        total = len(self.sdf_file_paths)
        train_split = int(self.split_ratio_t * total)
        valid_split = int(self.split_ratio_v * total)
        self.train_paths = self.sdf_file_paths[:train_split]
        self.valid_paths = self.sdf_file_paths[train_split:train_split + valid_split]
        self.test_paths = self.sdf_file_paths[train_split + valid_split:]

        with open(f'{self.cfg.logger.log_dir}/train_paths.log', 'w') as train_file:
            for path in self.train_paths:
                train_file.write(f"{path}\n")

        with open(f'{self.cfg.logger.log_dir}/valid_paths.log', 'w') as valid_file:
            for path in self.valid_paths:
                valid_file.write(f"{path}\n")

        with open(f'{self.cfg.logger.log_dir}/test_paths.log', 'w') as test_file:
            for path in self.test_paths:
                test_file.write(f"{path}\n")

    def compute_hash(self, paths):
        # Compute the hash of the list of paths
        hasher = hashlib.sha256()
        for path in sorted(paths):  # Sorting ensures consistent order
            hasher.update(path.encode('utf-8'))
        return hasher.hexdigest()

    def verify_splits(self):
        # Verify the hash of the paths to ensure consistency
        train_hash = self.compute_hash(self.train_paths)
        valid_hash = self.compute_hash(self.valid_paths)
        test_hash = self.compute_hash(self.test_paths)
        info(f"Train len: {len(self.train_paths)} hash: {train_hash}")
        info(f"Valid len: {len(self.valid_paths)} hash: {valid_hash}")
        info(f"Test len: {len(self.test_paths)} hash: {test_hash}")
        return train_hash, valid_hash, test_hash
    
    def copy_files_to_split_dirs(self):
        global copied
        if copied or self.cfg.train.train_tag == "s1_pretrain":
            return

        test_pos_dir = os.path.join(self.cfg.data.save_path, 'test', 'pos_cleaned_sequences_mol')
        test_neg_dir = os.path.join(self.cfg.data.save_path, 'test', 'neg_cleaned_sequences_mol')

        # if folder exists, return
        if os.path.exists(test_pos_dir) and os.path.exists(test_neg_dir):
            return

        os.makedirs(test_pos_dir)
        os.makedirs(test_neg_dir)

        # Copy files to respective directories
        def copy_files(paths, pos_dir, neg_dir):
            for path in paths:
                if "pos_cleaned_sequences_mol" in path:
                    shutil.copy(path, pos_dir)
                elif "neg_cleaned_sequences_mol" in path:
                    shutil.copy(path, neg_dir)

        copy_files(self.test_paths, test_pos_dir, test_neg_dir)

        info(f"Copied {len(self.test_paths)} files to test directories")

        copied = True

    def mol_to_heterograph(self, sdf_path):
        mol = read_SDF_file(sdf_path)
        if self.cfg.train.frag_method == "FraSCESS":
            # g: dgl.heterograph
            g = mol2heterograph_frascess(mol, sdf_path, self.vocab_dict)
        elif self.cfg.train.frag_method == "AdaFrag":
            from .Pepland.data import Mol2HeteroGraph as Pepland_Mol2HG
            # load pepland vocab list.
            g = Pepland_Mol2HG(mol)
        elif self.cfg.train.frag_method == "BRICS":
            g = mol2heterograph_BRICS(mol, sdf_path, self.vocab_dict)

        '''
        if g.number_of_nodes() > 0:
            if self.mask_graph and self.cfg.train.train_tag == "s1_pretrain":
                # for stage 1 pretraining, mask the graph nodes.
                return self.mask_graph(g)
            else:
                return g
        return None
        '''
        
        if g.number_of_nodes() > 0:
            # default label is "-1"
            label = -1
            if self.cfg.train.train_tag != "s1_pretrain":
                if self.cfg.train.train_tag != "eval":
                    # Assign a label based on the sdf_path content
                    if "pos_cleaned_sequences_mol" in sdf_path:
                        label = 1
                    elif "neg_cleaned_sequences_mol" in sdf_path:
                        label = 0
                    else:
                        raise ValueError(f"Unexpected sdf_path: {sdf_path}")
                else:
                    # Assign a label based on the sdf_path content, 4 class for tsne good looking.
                    if "eval/traindf" in sdf_path:
                        if "pos_cleaned_sequences_mol" in sdf_path:
                            label = 3
                        elif "neg_cleaned_sequences_mol" in sdf_path:
                            label = 2
                    elif "pos_cleaned_sequences_mol" in sdf_path:
                        label = 1
                    elif "neg_cleaned_sequences_mol" in sdf_path:
                        label = 0
                    else:
                        raise ValueError(f"Unexpected sdf_path: {sdf_path}")

            if self.mask_graph and self.cfg.train.train_tag == "s1_pretrain":
                # for stage 1 pretraining, mask the graph
                return self.mask_graph(g), label
            else:
                return g, label
        return None

    def load_vocabdict(self, vocabjson_path):
        with open(vocabjson_path, 'r') as f:
            data = json.load(f)
        
        frequent_fragments = data.get("frequent_fragments", {})
        vocab_dict = {fragment: i for i, (fragment, count) in enumerate(frequent_fragments.items())}

        info(f"Loaded {len(vocab_dict)} fragments from {vocabjson_path}")

        # info("Vocab dict:", vocab_dict) {0: 'NCC=O', 1: 'CC=O', 2: 'CC(N)=O'...
        
        self.vocab_dict = vocab_dict

    def slice_path(self, path) -> list:
        # slice the path by "+".
        return path.split(" + ")
    
    def _get_paths_for_stage(self, stage):
        if stage == 'train':
            return self.train_paths
        elif stage == 'valid':
            return self.valid_paths
        elif stage == 'test':
            return self.test_paths
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # Single-process data loading
            paths = self._get_paths_for_stage(self.stage)
        else:  # Multi-process data loading
            # Split the dataset into chunks according to the number of workers
            total_workers = worker_info.num_workers
            worker_id = worker_info.id

            paths = self._get_paths_for_stage(self.stage)
            
            # Split the paths among the workers
            per_worker = int(math.ceil(len(paths) / total_workers))
            worker_paths = paths[worker_id * per_worker:(worker_id + 1) * per_worker]

        for path in worker_paths:
            graph = self.mol_to_heterograph(path)
            if graph is not None:
                yield graph

    def __len__(self):
        return len(self._get_paths_for_stage(self.stage))

    def test_loader(self):
        info("Testing DataLoader...")
        self.load_vocabdict("../tokenizer/vocablist/vocab_1.json")
        self.load_SDF_directory(self.cfg.data.dataset_path)
        graphs = self.mol_to_heterograph()
        info(f"Number of graphs: {len(graphs)}")
        exit()

def collate_fn(batch):
    graphs, labels = map(list, zip(*batch))
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels, dtype=torch.float32)
    return batched_graph, labels

def create_dataset(cfg, stage, mask_graph, split_ratio_t, split_ratio_v):
    info(f"Creating dataset for {stage}...")
    return MoleculeDataLoader(cfg, stage, mask_graph, split_ratio_t, split_ratio_v)

def make_loaders(cfg, batch_size, mask_graph, split_ratio_t = 0.8, split_ratio_v=0.1):
    if cfg.train.train_tag == "eval":
        # For evaluation, we only need the test dataset
        info("Creating test dataset for evaluation...")
        test_dataset = create_dataset(cfg, "test", mask_graph, split_ratio_t, split_ratio_v)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=cfg.data.num_workers, collate_fn=collate_fn)
        dataloaders = {}
        dataloaders["test"] = test_loader
        return dataloaders, test_dataset.vocab_dict

    train_dataset = create_dataset(cfg, "train", mask_graph, split_ratio_t, split_ratio_v)
    valid_dataset = create_dataset(cfg, "valid", mask_graph, split_ratio_t, split_ratio_v)
    test_dataset = create_dataset(cfg, "test", mask_graph, split_ratio_t, split_ratio_v)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=cfg.data.num_workers, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=cfg.data.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=cfg.data.num_workers, collate_fn=collate_fn)

    # train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, num_workers=cfg.data.num_workers)
    # valid_loader = GraphDataLoader(valid_dataset, batch_size=batch_size, num_workers=cfg.data.num_workers)
    # test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, num_workers=cfg.data.num_workers)

    dataloaders = {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader
    }

    return dataloaders, train_dataset.vocab_dict

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
            # info(bond_props)
            for prop in bond_props:
                idx, value = prop.split(':')
                bond = mol.GetBondWithIdx(int(idx))
                bond.SetProp("LinkingBond", value)
        break  # we only want the first molecule in the file

    if mol is None:
        raise ValueError("No valid molecule found in the SDF file")
    
    return mol