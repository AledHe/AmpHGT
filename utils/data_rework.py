"""
Note that some code in this script is adapted or modified from PepLand and PharmHGT.
This dataloader will only read one SDF file.
Discarded the split function.
"""

import json
import os
import random
import lz4.frame

import torch
import pickle
import dgl

from .FraSCESS.mol2heterograph_FraSCESS import mol2heterograph_frascess
from .RegularMethod.BRICSFrag import mol2heterograph_BRICS

from rdkit import Chem
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Iterator, Tuple
from dgl import DGLGraph
from utils.std_logger import error, info, warning
from tqdm import tqdm

# Each of our graphs is not very large. However, we have many graphs (for the pre-training step), 
# so it seems unnecessary to worry about the shuffle method, 
# because we should stop training before putting all the data into the stage, maybe for 10k steps or so, 
# because the performance may not improve anymore, and IterableDataset is enough to meet the needs. 
# Therefore, we might keep the current dataset and use it as the loader of the fine-tune dataset. 
# The fine-tune dataset can be all loaded into the memory. But any improvement is welcome.

#TODO: implement a new streaming dataloader class for pretraining.
#TODO: implement label logic for regression tasks. (Solub, MIC)

class PretrainDataLoader(IterableDataset):
    """
    Streaming loading data for pretraining stage

    This function needs implementation.
    It should support IterableDataset with loading graphs.
    This class will not generate graphs, only load them from the cache.
    each lz4 file contains a chunk of graphs, 2000 here by default.
    does IterableDataset know how many graphs we've processed?
    if it can, we can always load 1 chunk ahead of the current chunk (the processing one) for efficiency.
    """
    def __init__(self, cfg, stage: str):
        super().__init__()
        self.cfg = cfg
        self.stage = stage

        sdf_basename = os.path.splitext(os.path.basename(self.split_path(stage)))[0]
        self.cache_dir = os.path.join(self.cfg.data.tmp_dir, f"{self.stage}_{sdf_basename}")

        # Validate cache directory
        if not os.path.isdir(self.cache_dir):
            raise ValueError(f"Invalid cache directory: {self.cache_dir}")
        
        # Collect chunk files and shuffle if training
        self.chunk_files = sorted([
            os.path.join(self.cache_dir, f) 
            for f in os.listdir(self.cache_dir) 
            if f.endswith('.lz4')
        ])
        if not self.chunk_files:
            raise RuntimeError(f"No .lz4 chunks found in {self.cache_dir}")
        
        if self.stage == "train":
            random.shuffle(self.chunk_files)  # In-place shuffle for training
            
        # Load vocabulary if provided
        self.vocab_dict = None
        if cfg.data.vocablist:
            self.vocab_dict = load_vocabdict(cfg.data.vocablist)
            info(f"Loaded vocab with {len(self.vocab_dict)} items")
        else:
            warning("Pretraining without vocabulary dictionary")
            
    def split_path(self, stage):
        if stage == "train":
            return self.cfg.data.train_sdf
        elif stage == "valid":
            return self.cfg.data.valid_sdf
        elif stage == "test":
            return self.cfg.data.test_sdf
        elif stage == "generate_vocabdict":
            return self.cfg.data.dataset
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
    def _load_chunk(self, chunk_path: str) -> Iterator[Tuple[DGLGraph, int]]:
        """Load and decompress a single chunk file"""
        try:
            # Read and decompress
            bin_file = chunk_path.replace(".lz4", "")
            with open(chunk_path, 'rb') as f_in:
                with open(bin_file, 'wb') as f_out:
                    f_out.write(lz4.frame.decompress(f_in.read()))
            
            # Load graphs and labels
            graphs, labels_dict = dgl.load_graphs(bin_file)
            os.remove(bin_file)
            labels = labels_dict["labels"].tolist()
            
            # Validate matching counts
            if len(graphs) != len(labels):
                warning(f"Chunk {os.path.basename(chunk_path)} has {len(graphs)} graphs but {len(labels)} labels")
                return
            
            # yield, no need for masking, the graph is saved with masked nodes.
            for g, lbl in zip(graphs, labels):
                yield g, lbl
                
        except Exception as e:
            error(f"Error loading chunk {chunk_path}: {str(e)}")
        
    def __iter__(self) -> Iterator[Tuple[DGLGraph, int]]:
        """Main iteration logic with worker-aware chunk distribution"""
        worker_info = torch.utils.data.get_worker_info()
        
        # Split chunks among workers
        if worker_info is None:  # Single-process
            my_chunks = self.chunk_files
        else:  # Multi-worker
            per_worker = len(self.chunk_files) // worker_info.num_workers
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = start_idx + per_worker if worker_id < worker_info.num_workers - 1 else None
            my_chunks = self.chunk_files[start_idx:end_idx]
            info(f"Worker {worker_id} processing {len(my_chunks)} chunks")

        # Stream chunks
        for chunk in my_chunks:
            yield from self._load_chunk(chunk)  # Delegate to chunk loader
        
class FinetuneDataLoader(Dataset):
    """
    Dataloader for fine-tune
    Will discard mask_graph param soon.
    """
    def __init__(self, cfg, stage, mask_graph, sdf_path=None):
        # sdf_path is made for vocab gen.
        self.cfg = cfg
        self.stage = stage
        self.mask_graph_fn = mask_graph
        self.vocab_dict = None
        if self.cfg.logger.test_run:
            info("Test run enabled.")
        if cfg.data.vocablist:
            # Load vocabulary
            self.vocab_dict = load_vocabdict(cfg.data.vocablist)
        else:
            warning("No vocabulary list provided.")
            self.vocab_dict = None

        # Gather SDF paths for the given stage
        #  - e.g., train -> [train_sdf_pos, train_sdf_neg]
        #  - e.g., valid -> [valid_sdf_pos, valid_sdf_neg]
        #  - e.g., test  -> [test_sdf_pos, test_sdf_neg]
        self.sdf_paths = self.split_path(stage)

        if cfg.train.train_tag == "generate_vocabdict":
            info("Vocab generation mode. Skipping graph processing.")
            # Just load raw molecules (SMILES) for the entire set
            self.mols = []
            self.mols.extend(read_SMI_file(sdf_path))
            self.graphs_labels = []
        else:
            # Pre-process all SDF files into (graph, label) pairs
            self.graphs_labels = []
            for sdf_path in self.sdf_paths:
                self.graphs_labels.extend(self.mol_to_heterograph(sdf_path))

    def split_path(self, stage):
        """
        Returns a list of SDF file paths for the given stage.
        The config is expected to define the fields:
            - train_sdf_pos, train_sdf_neg
            - valid_sdf_pos, valid_sdf_neg
            - test_sdf_pos, test_sdf_neg
        """
        if stage == "train":
            return [self.cfg.data.train_sdf_pos, self.cfg.data.train_sdf_neg]
        elif stage == "valid":
            return [self.cfg.data.valid_sdf_pos, self.cfg.data.valid_sdf_neg]
        elif stage == "test":
            return [self.cfg.data.test_sdf_pos, self.cfg.data.test_sdf_neg]
        elif stage == "generate_vocabdict":
            return [self.cfg.data.dataset]
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def mol_to_heterograph(self, sdf_path):
        """
        For a single SDF file path, this method:
          - Checks cache,
          - If no cache, processes molecules into DGLGraphs,
          - Returns a list of (graph, label) tuples.

        We keep the original chunk-based caching approach,
        but we store each SDF path in a separate sub-directory to avoid conflicts.
        """
        # Make a unique cache directory for each SDF file
        # so that positive/negative caches don't conflict
        sdf_basename = os.path.splitext(os.path.basename(sdf_path))[0]
        cache_dir = os.path.join(self.cfg.data.tmp_dir, f"{self.stage}_{sdf_basename}")
        os.makedirs(cache_dir, exist_ok=True)

        # Check if cached graphs are available
        cache_files = sorted(os.listdir(cache_dir))
        if cache_files:
            info(f"Loading cached graphs from {cache_dir}...")
            return load_graphs(cache_dir)

        # If no cache exists, process the graphs and save them
        info(f"No cached graphs found in {cache_dir}. Processing {sdf_path} and caching results...")
        self.save_cache(sdf_path, self.vocab_dict, cache_dir)
        return load_graphs(cache_dir)
    
    def save_cache(self, sdf_path, vocab_dict, cache_dir, chunk_size=500):
        """
        Save graphs to local cache in chunks to avoid recomputing graphs every time.
        Each chunk contains up to `chunk_size` graphs.

        Args:
        - sdf_path (str): Path to the SDF file containing molecules.
        - vocab_dict (dict): Vocabulary of molecular fragments.
        - cache_dir (str): Directory to store the graph cache.
        - chunk_size (int): Number of graphs to save in each chunk.
        """
        # Read molecules from SDF file
        # Note that we've switched to SMILES storage.
        mols = read_SMI_file(sdf_path)

        # Process molecules in chunks
        chunk_idx = 0
        graphs_batch = []
        labels_batch = []

        # if cfg defined chunk_size, use it, otherwise use the default value
        if self.cfg.data.chunk_size:
            chunk_size = self.cfg.data.chunk_size

        for idx, mol in enumerate(tqdm(mols, desc="Processing molecules", unit="mol")):
            # Create graph for the molecule based on selected fragmentation method
            if self.cfg.train.frag_method == "FraSCESS":
                g = mol2heterograph_frascess(mol, mol.GetProp("_Name"), vocab_dict)
            elif self.cfg.train.frag_method == "AdaFrag":
                from .Pepland.data import Mol2HeteroGraph as Pepland_Mol2HG
                g = Pepland_Mol2HG(mol)
            elif self.cfg.train.frag_method == "BRICS":
                g = mol2heterograph_BRICS(mol, mol.GetProp("_Name"), vocab_dict)
            else:
                raise ValueError(f"Unknown fragmentation method: {self.cfg.train.frag_method}")
            
            if g.number_of_nodes() > 0:
                label = self.infer_label(sdf_path)
                if self.mask_graph_fn and self.cfg.train.train_tag == "s1_pretrain":
                    self.mask_graph_fn(g) # Fixed bugs. shouldn't use g = self.mask_graph_fn(g) for mask_graph_fn(g) returns nothing.
            else:
                warning(f"Molecule {mol.GetProp('_Name')} with no nodes.")
                continue

            graphs_batch.append(g)
            labels_batch.append(label)

            # If the batch size reaches chunk_size, save it to disk
            if len(graphs_batch) >= chunk_size:
                chunk_filename = os.path.join(cache_dir, f"graphs_{self.stage}_chunk_{chunk_idx}.bin")

                save_labels = {
                    "labels": torch.tensor(labels_batch),
                }

                dgl.save_graphs(chunk_filename, graphs_batch, save_labels)
                # Compress the .bin file using LZ4
                with open(chunk_filename, 'rb') as f_in:
                    with open(chunk_filename + '.lz4', 'wb') as f_out:
                        f_out.write(lz4.frame.compress(f_in.read(), compression_level=8))
                os.remove(chunk_filename)
                info(f"Saved chunk {chunk_idx} with {len(graphs_batch)} graphs.")
                chunk_idx += 1
                graphs_batch = []  # Reset batch for next chunk
                labels_batch = []

        # Save any remaining graphs that didn't reach chunk_size
        if graphs_batch:
            chunk_filename = os.path.join(cache_dir, f"graphs_{self.stage}_chunk_{chunk_idx}.bin")

            save_labels = {
                "labels": torch.tensor(labels_batch),
            }

            dgl.save_graphs(chunk_filename, graphs_batch, save_labels)
            # Compress the .bin file using LZ4
            with open(chunk_filename, 'rb') as f_in:
                with open(chunk_filename + '.lz4', 'wb') as f_out:
                    f_out.write(lz4.frame.compress(f_in.read(), compression_level=8))
            os.remove(chunk_filename)
            info(f"Saved final chunk {chunk_idx} with {len(graphs_batch)} graphs.")
    
    def infer_label(self, sdf_path):
        """
        Determine the label for a molecule based on the SDF file path.
        """
        if "unlabel" in sdf_path:
            return -1
        elif "positive" in sdf_path:
            return 1
        elif "negative" in sdf_path:
            return 0
        elif self.cfg.logger.test_run:
            return -1
        else:
            raise ValueError("Invalid SDF file name or labeling logic not handled.")

    def __getitem__(self, idx):
        return self.graphs_labels[idx]

    def __len__(self):
        return len(self.graphs_labels)
    
class FinetuneDataLoader_RG(Dataset):
    """
    Load Regression Data.
    """
    def  __init__(self, cfg, stage, sdf_path=None):
        self.cfg = cfg
        self.stage = stage
        self.vocab_dict = None
        if self.cfg.logger.test_run:
            info("Test run enabled.")
        if cfg.data.vocablist:
            # Load vocabulary
            self.vocab_dict = load_vocabdict(cfg.data.vocablist)
        else:
            warning("No vocabulary list provided.")
            self.vocab_dict = None

        self.sdf_paths = self.split_path(stage)
        self.csv_path = cfg.train.csv_path

        self.sequences_value = self.load_csv_data() # for look ups.

        self.fasta_dict = self.load_fasta()

        self.graphs_regs_labels = []
        for sdf_path in self.sdf_paths:
            self.graphs_regs_labels.extend(self.mol_to_heterograph(sdf_path))

    def mol_to_heterograph(self, sdf_path):
        """
        Similar to the method in FinetuneDataLoader
        """
        # Make a unique cache directory for each SDF file
        # so that positive/negative caches don't conflict
        sdf_basename = os.path.splitext(os.path.basename(sdf_path))[0]
        cache_dir = os.path.join(self.cfg.data.tmp_dir, f"RG_{self.stage}_{sdf_basename}")
        os.makedirs(cache_dir, exist_ok=True)

        # Check if cached graphs are available
        cache_files = sorted(os.listdir(cache_dir))
        if cache_files:
            info(f"Loading cached graphs from {cache_dir}...")
            return load_graphs(cache_dir)
        
        # If no cache exists, process the graphs and save them
        info(f"No cached graphs found in {cache_dir}. Processing {sdf_path} and caching results...")
        self.save_cache(sdf_path, self.vocab_dict, cache_dir)
        return load_graphs(cache_dir)
    
    def save_cache(self, sdf_path, vocab_dict, cache_dir, chunk_size=500):
        """
        Similar to the method in FinetuneDataLoader
        """
        # Read molecules from SDF file
        # Note that we've switched to SMILES storage.
        mols = read_SMI_file(sdf_path)

        # Process molecules in chunks
        chunk_idx = 0
        graphs_batch = []
        values_batch = []

        if self.cfg.data.chunk_size:
            chunk_size = self.cfg.data.chunk_size

        # if cfg defined chunk_size, use it, otherwise use the default value
        for idx, mol in enumerate(tqdm(mols, desc="Processing molecules", unit="mol")):
            # retrieve fasta seq using mol.GetProp("_Name")
            fasta_seq = self.fasta_dict.get(mol.GetProp("_Name"), None)
            # use fasta_seq to look up the sequence in the csv file
            reg_value = self.sequences_value.get(fasta_seq, None)
            # Create graph for the molecule based on selected fragmentation method
            if self.cfg.train.frag_method == "FraSCESS":
                g = mol2heterograph_frascess(mol, mol.GetProp("_Name"), vocab_dict)
            elif self.cfg.train.frag_method == "AdaFrag":
                from .Pepland.data import Mol2HeteroGraph as Pepland_Mol2HG
                g = Pepland_Mol2HG(mol)
            elif self.cfg.train.frag_method == "BRICS":
                g = mol2heterograph_BRICS(mol, mol.GetProp("_Name"), vocab_dict)
            else:
                raise ValueError(f"Unknown fragmentation method: {self.cfg.train.frag_method}")
            
            if g.number_of_nodes() > 0:
                if self.mask_graph_fn and self.cfg.train.train_tag == "s1_pretrain":
                    self.mask_graph_fn(g) # Fixed bugs. shouldn't use g = self.mask_graph_fn(g) for mask_graph_fn(g) returns nothing.
            else:
                warning(f"Molecule {mol.GetProp('_Name')} with no nodes.")
                continue

            graphs_batch.append(g)
            values_batch.append(reg_value)

            # If the batch size reaches chunk_size, save it to disk
            if len(graphs_batch) >= chunk_size:
                chunk_filename = os.path.join(cache_dir, f"graphs_{self.stage}_chunk_{chunk_idx}.bin")

                save_values = {
                    "values": torch.tensor(values_batch),
                }

                dgl.save_graphs(chunk_filename, graphs_batch, save_values)
                # Compress the .bin file using LZ4
                with open(chunk_filename, 'rb') as f_in:
                    with open(chunk_filename + '.lz4', 'wb') as f_out:
                        f_out.write(lz4.frame.compress(f_in.read(), compression_level=8))
                os.remove(chunk_filename)
                info(f"Saved chunk {chunk_idx} with {len(graphs_batch)} graphs.")
                chunk_idx += 1
                graphs_batch = []  # Reset batch for next chunk
                values_batch = []

        # Save any remaining graphs that didn't reach chunk_size
        if graphs_batch:
            chunk_filename = os.path.join(cache_dir, f"graphs_{self.stage}_chunk_{chunk_idx}.bin")

            save_values = {
                "values": torch.tensor(values_batch),
            }

            dgl.save_graphs(chunk_filename, graphs_batch, save_values)
            # Compress the .bin file using LZ4
            with open(chunk_filename, 'rb') as f_in:
                with open(chunk_filename + '.lz4', 'wb') as f_out:
                    f_out.write(lz4.frame.compress(f_in.read(), compression_level=8))
            os.remove(chunk_filename)
            info(f"Saved final chunk {chunk_idx} with {len(graphs_batch)} graphs.")

    def split_path(self, stage):
        """
        Returns a list of SDF file paths for the given stage.
        The config is expected to define the fields:
            - train_sdf_pos, train_sdf_neg
            - valid_sdf_pos, valid_sdf_neg
            - test_sdf_pos, test_sdf_neg
        """
        if stage == "train":
            return [self.cfg.data.train_sdf_pos, self.cfg.data.train_sdf_neg]
        elif stage == "valid":
            return [self.cfg.data.valid_sdf_pos, self.cfg.data.valid_sdf_neg]
        elif stage == "test":
            return [self.cfg.data.test_sdf_pos, self.cfg.data.test_sdf_neg]
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
    def load_csv_data(self):
        import pandas as pd
        df = pd.read_csv(self.csv_path)
        
        df = df.dropna(subset=["SEQUENCE", "μM"])
        sequence_reg_dict = dict(zip(df["SEQUENCE"], df["μM"].astype(float)))
        
        return sequence_reg_dict
    
    def load_fasta(self):
        """
        load all fasta in the fasta folder for sequence/title lookup.
        """
        # find all fasta files in the fasta folder, get path
        fasta_files = [os.path.join("fasta_RG", f) for f in os.listdir("fasta_RG") if f.endswith(".fasta")]
        # read all fasta files into a dict
        fasta_dict = {}
        for fasta_file in fasta_files:
            fasta_dict.update(read_fasta(fasta_file))
        return fasta_dict
    
    def __getitem__(self, idx):
        graph, reg_value, label = self.graphs_regs_labels[idx]
        return graph, reg_value, label

    def __len__(self):
        return len(self.graphs_regs_labels)
    
def load_vocabdict(vocabjson_path):
    with open(vocabjson_path, 'r') as f:
        data = json.load(f)
    
    frequent_fragments = data.get("frequent_fragments", {})
    vocab_dict = {
        fragment: i
        for i, (fragment, count) in enumerate(frequent_fragments.items())
    }

    info(f"Loaded {len(vocab_dict)} fragments from {vocabjson_path}")

    # info("Vocab dict:", vocab_dict) {0: 'NCC=O', 1: 'CC=O', 2: 'CC(N)=O'...}

    return vocab_dict

def collate_fn(batch):
    graphs, labels = zip(*batch) # Unzip the batch into two lists
    batched_graph = dgl.batch(list(graphs)) # Combine all the graphs into a single batch
    labels = torch.tensor(list(labels), dtype=torch.float32) # Combine all labels into a tensor
    return batched_graph, labels

def collate_fn_with_reg(batch):
    graphs, values = zip(*batch)
    batched_graph = dgl.batch(list(graphs))
    values = torch.tensor(list(values), dtype=torch.float32)
    return batched_graph, values

def make_pretrain_loaders(cfg, batch_size):
    # 1) Create datasets for each split
    train_dataset = PretrainDataLoader(cfg, "train")

    if cfg.logger.test_run:
        valid_dataset = train_dataset
        test_dataset = train_dataset
    else:
        valid_dataset = PretrainDataLoader(cfg, "valid")
        test_dataset = PretrainDataLoader(cfg, "test")

    # 2) Wrap them in PyTorch DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    dataloaders = {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader
    }
    
    return dataloaders, train_dataset.vocab_dict

def make_binary_loaders(cfg, batch_size, mask_graph, inference=False):
    if not inference:
        # 1) Create datasets for each split
        train_dataset = FinetuneDataLoader(cfg, "train", mask_graph)
        
        if cfg.logger.test_run:
            valid_dataset = train_dataset
            test_dataset = train_dataset
        else:
            valid_dataset = FinetuneDataLoader(cfg, "valid", mask_graph)
            test_dataset  = FinetuneDataLoader(cfg, "test", mask_graph)

        # 2) Wrap them in PyTorch DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            shuffle=True
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

    else:
        info("Inference mode. Loading test dataset only.")
        test_dataset = FinetuneDataLoader(cfg, "test", mask_graph)

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=cfg.data.num_workers,
            pin_memory=True
        )

        dataloaders = {
            "test": test_loader
        }
        
    return dataloaders, train_dataset.vocab_dict if not inference else test_dataset.vocab_dict

def make_regression_loaders(cfg, batch_size, inference=False):
    if not inference:
        # 1) Create datasets for each split
        train_dataset = FinetuneDataLoader_RG(cfg, "train")
        
        if cfg.logger.test_run:
            valid_dataset = train_dataset
            test_dataset = train_dataset
        else:
            valid_dataset = FinetuneDataLoader_RG(cfg, "valid")
            test_dataset  = FinetuneDataLoader_RG(cfg, "test")

        # 2) Wrap them in PyTorch DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn_with_reg,
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            shuffle=True
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn_with_reg,
            num_workers=cfg.data.num_workers,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn_with_reg,
            num_workers=cfg.data.num_workers,
            pin_memory=True
        )

        # 3) Return the loaders plus the vocabulary from one of them (or a consistent place)
        dataloaders = {
            "train": train_loader,
            "valid": valid_loader,
            "test": test_loader
        }

    else:
        info("Inference mode. Loading test dataset only.")
        test_dataset = FinetuneDataLoader_RG(cfg, "test")

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn_with_reg,
            num_workers=cfg.data.num_workers,
            pin_memory=True
        )

        dataloaders = {
            "test": test_loader
        }
        
    return dataloaders, train_dataset.vocab_dict

def load_graphs(cache_dir):
    """
    Load all graph chunks from the cache directory.

    Args:
    - cache_dir (str): Directory containing the saved graph chunks.

    Returns:
    - List of (graph, label) tuples
    """
    graphs_labels = []

    # Iterate through all chunk files in the cache directory
    for filename in sorted(os.listdir(cache_dir)):
        if filename.endswith(".lz4"):
            filepath = os.path.join(cache_dir, filename)
            
            # Decompress the .lz4 file back to .bin
            bin_file = filepath.replace(".lz4", "")
            with open(filepath, 'rb') as f_in:
                with open(bin_file, 'wb') as f_out:
                    f_out.write(lz4.frame.decompress(f_in.read()))

            graphs, labels_dict = dgl.load_graphs(bin_file) # returns graph_list: list[DGLGraph], labels: dict[str, Tensor]
            os.remove(bin_file)
            labels = labels_dict["labels"].tolist()
            fasta_seq_loaded = pickle.loads(bytes(labels_dict["fasta_pickle"].tolist()))
            graphs_labels.extend(zip(graphs, labels, fasta_seq_loaded)) # Combine graphs and labels into tuples

    return graphs_labels

def unzip_atom_residue_mapping(indices_str: str) -> list[int]:
    indices = []
    for part in indices_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    return indices

def read_SMI_file(file_path: str) -> list[Chem.Mol]:
    """
    Read multiple mols from a single SMILES file, and extract bond properties.
    
    :param file_path: Path to the SMILES file
    :return: List of RDKit Mol objects with bond properties restored
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File path does not exist. {file_path}")
    
    suppl = Chem.SmilesMolSupplier(file_path, titleLine=True, delimiter = '!^!') 
    mols = []
    
    for mol in suppl: 
        if mol is None:
            continue
        if mol.HasProp("_BondProps"):
            bond_props = mol.GetProp("_BondProps").split('|')
            # handle if not bond_props, this happens when a pseudo peptides with no sidechain, for example. Gly-Gly-Gly.
            # skip the molecule if no bond_props, This is a temp option for vocab generation. disable when training.
            if bond_props == ['']:
                pass
            else:
                for prop in bond_props:
                    idx, value = prop.split(':')
                    bond = mol.GetBondWithIdx(int(idx))
                    bond.SetProp("LinkingBond", value)
        if mol.HasProp("_AmideBonds"):
            amide_bonds = mol.GetProp("_AmideBonds").split('|')
            # skip the molecule if no bond_props, This is a temp option for vocab generation. disable when training.
            if amide_bonds == ['']:
                pass
            else:
                for prop in amide_bonds:
                    idx, value = prop.split(':')
                    bond = mol.GetBondWithIdx(int(idx))
                    bond.SetProp("AmideBond", value)
        if mol.HasProp("_Amino_Label"):
            aac_label_str = mol.GetProp("_Amino_Label")
            for aac_entry in aac_label_str.split(';'):
                try:
                    aac_part, indices_info = aac_entry.split(':[', 1)
                    indices_str = indices_info.rstrip(']')
                    
                    indices = unzip_atom_residue_mapping(indices_str)
                    
                    for idx in indices:
                        atom = mol.GetAtomWithIdx(idx)
                        atom.SetProp("aac", aac_part)
                        
                except Exception as e:
                    print(f"Error parsing AAC entry '{aac_entry}': {str(e)}")
                    continue
        mols.append(mol)

    if not mols:
        raise ValueError("No valid molecule found in the SMILES file")
    
    info(f"Read {len(mols)} molecules from {file_path}")
    
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
    
    info(f"Read {len(mols)} molecules from {file_path}")
    
    return mols

def read_fasta(fastapath):
    """
    Read fasta file and return a list of sequences with titles.
    """
    sequences = []
    with open(fastapath, 'r', encoding="utf-8") as f:
        title = None
        seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if title:
                    sequences.append((title, ''.join(seq)))
                title = line[1:]
                seq = []
            else:
                seq.append(line)
        if title:
            sequences.append((title, ''.join(seq)))
    return sequences