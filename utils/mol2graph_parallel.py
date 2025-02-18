"""
This script is used to generate chunk graph data with multiprocessing.
"""

import math
import os
import multiprocessing

import lz4.frame
import torch
import dgl
from rdkit import Chem
from tqdm import tqdm

try:
    from .FraSCESS.mol2heterograph_FraSCESS import mol2heterograph_frascess
    from .RegularMethod.BRICSFrag import mol2heterograph_BRICS
    # from .Pepland.data import Mol2HeteroGraph as Pepland_Mol2HG
    from .data_rework import load_vocabdict, read_SMI_file, read_SDF_file
    from .seed_device import generate_random_string
except ImportError as e:
    print(f"Error importing fragmentation methods: {e}. Please ensure the script is run from the correct location or adjust import paths.")
    raise

# Set RDKit default pickle properties
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
torch.multiprocessing.set_sharing_strategy('file_system')

# Global variables for multiprocessing
global_vocab_dict = None
global_frag_method = None
global_cfg = None

#TODO: Design worker batch return to avoid pickle overhead. IMPORTANT.
# It has observed that single tread processing is equvalent to multi-thread processing.
# my guess is that the data pickle and send back to main process is time-consuming.
# can we: for each worker, it won't pickle the g and label back instead directly save to disk when reached chunk_size?
# DONE, speed is considerable improved.

#TODO: get_fragment_feats should be optimized

def init_process(vocab_dict, frag_method, cfg):
    """
    Initializes global variables for each worker process.
    This function is called when each process starts.
    """
    global global_vocab_dict
    global global_frag_method
    global global_cfg
    global_vocab_dict = vocab_dict
    global_frag_method = frag_method
    global_cfg = cfg

def mol_to_heterograph_standalone(mol: Chem.Mol):
    """Converts a single RDKit Mol object to a DGL heterograph based on the fragmentation method."""
    if global_frag_method == "FraSCESS":
        g = mol2heterograph_frascess(mol, mol.GetProp("_Name"), global_vocab_dict)
    elif global_frag_method == "AdaFrag":
        g = Pepland_Mol2HG(mol)
    elif global_frag_method == "BRICS":
        g = mol2heterograph_BRICS(mol, mol.GetProp("_Name"), global_vocab_dict)
    else:
        raise ValueError(f"Unknown fragmentation method: {global_frag_method}")
    return g

def process_mol(mol):
    """
    Processes a single molecule to (graph, label).
    If something goes wrong, raises an exception.
    """
    g = mol_to_heterograph_standalone(mol)
    if g.number_of_nodes() == 0:
        raise ValueError(f"Molecule {mol.GetProp('_Name')} has no nodes.")
    # Example: label = -1 for pretrain
    label = -1
    return g, label
    
def bin_save(batch, cache_dir, stage, chunk_idx):
    graphs = [g for g, _ in batch]
    labels = torch.tensor([l for _, l in batch])

    chunk_filename = os.path.join(cache_dir, f"graphs_{stage}_chunk_{chunk_idx}.bin")
    tmp_path = f"{chunk_filename}.tmp"
    
    dgl.save_graphs(tmp_path, graphs, {"labels": labels})
    os.rename(tmp_path, chunk_filename)

    # Just do the compression synchronously
    with open(chunk_filename, 'rb') as f_in:
        with open(chunk_filename + '.lz4', 'wb') as f_out:
            f_out.write(lz4.frame.compress(f_in.read(), compression_level=8))

    os.remove(chunk_filename)
    print(f"Info: Saved and compressed chunk {chunk_idx} with {len(batch)} graphs.")

def process_chunk_of_mols(args):
    """
    Each worker receives an entire chunk of molecules and saves them directly.
    
    Returns only the chunk index (or some small success flag).
    """
    chunk_idx, chunk_data, cache_sub_dir, stage, mask_graph = args
    
    current_batch = []
    for mol in chunk_data:
        try:
            g, label = process_mol(mol)
            if mask_graph and global_cfg.train.train_tag == "s1_pretrain":
                mask_graph(g)
            current_batch.append((g, label))
        except Exception as e:
            print(f"Error in chunk {chunk_idx}, mol {mol.GetProp('_Name')}: {e}", flush=True)
            raise
    
    if len(current_batch) > 0:
        bin_save(current_batch, cache_sub_dir, stage, chunk_idx)
    return chunk_idx  # minimal object to return

def chunked_iterable(iterable, chunk_size):
    """
    A small helper generator that yields lists of size <= chunk_size.
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == chunk_size:
            yield batch
            batch = []
    # Yield the remainder
    if batch:
        yield batch

def single_process_test(cfg, mol_data, vocab_dict, cache_dir, stage, frag_method, mask_graph):
    """
    Single-process version for debugging and testing.
    """
    init_process(vocab_dict, frag_method, cfg)
    chunk_idx = 0
    for chunk_data in chunked_iterable(mol_data, cfg.data.chunk_size):
        process_chunk_of_mols((chunk_idx, chunk_data, cache_dir, stage, mask_graph))
        chunk_idx += 1
        exit()

def save_cache_standalone(cfg, mol_data, vocab_dict, cache_dir, stage, frag_method,
                          chunk_size=500, num_workers=None, mask_graph=None, sdf_path=None):
    """
    Uses a chunked approach: each worker processes and saves an entire chunk 
    to disk, returning only minimal metadata (the chunk index) to the main process.
    """
    sdf_basename = os.path.splitext(os.path.basename(sdf_path))[0]
    cache_sub_dir = os.path.join(cfg.data.tmp_dir, f"{stage}_{sdf_basename}")
    os.makedirs(cache_sub_dir, exist_ok=True)

    # single_process_test(cfg, mol_data, vocab_dict, cache_sub_dir, stage, frag_method, mask_graph)

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()

    # calculate line in sdf file
    if sdf_path:
        with open(sdf_path, 'r', encoding="utf-8") as f:
            total_mols = sum(1 for line in f if line.strip()) - 1

    # Force 'spawn' method for better isolation
    multiprocessing.set_start_method('spawn', force=True)

    # Create the Pool
    with multiprocessing.Pool(
        processes=num_workers,
        initializer=init_process,
        initargs=(vocab_dict, frag_method, cfg)
    ) as pool:

        # We chunk the mol_data generator
        chunk_args_iter = (
            (chunk_idx, chunk, cache_sub_dir, stage, mask_graph)
            for chunk_idx, chunk in enumerate(chunked_iterable(mol_data, chunk_size))
        )

        results = []
        for res in tqdm(
            pool.imap_unordered(process_chunk_of_mols, chunk_args_iter),
            desc="Processing chunks",
            total = math.ceil(total_mols / chunk_size)
        ):
            # res is the chunk_idx that just finished
            results.append(res)

    print(f"Info: All {len(results)} chunks processed and saved to {cache_sub_dir}.")

def mol2graph(cfg, args, mask_graph, sdf_path):
    vocab_dict = load_vocabdict(cfg.data.vocablist)

    file_type = os.path.splitext(sdf_path)[1].lstrip('.').lower()

    if file_type == "smi" or file_type == "smiles":
        mol_data = read_SMI_file_generator(sdf_path)
    else:
        raise ValueError(f"Unsupported input file type: {file_type}. Please use SDF or SMILES files.")

    save_cache_standalone(cfg,
                          mol_data, 
                          vocab_dict, 
                          cfg.data.tmp_dir, 
                          args.stage, 
                          cfg.train.frag_method, 
                          cfg.data.chunk_size, 
                          args.num_workers,
                          mask_graph,
                          sdf_path)
    print(f"Info: Graph chunk processing complete. Caches saved.")

def read_SMI_file_generator(file_path: str):
    """
    Read the SMILES file line by line, parse it and return a generator.
    Each iteration returns an RDKit Mol object with the corresponding Bond attributes.
    File format example:
        SMILES!^!_Name!^!_BondProps!^!_AmideBonds

    :param file_path: Path to the SMILES file
    :return: Generator of RDKit Mol objects
    """
    if not os.path.exists(file_path):
        raise ValueError(f"File path does not exist: {file_path}")

    with open(file_path, 'r', encoding="utf-8") as f:
        header = next(f, None)

        for line_idx, line in enumerate(f, start=2):  # starting from line 2
            line = line.strip()
            if not line:
                continue

            splitted = line.split('!^!')
            if len(splitted) < 2:
                raise ValueError(f"Line {line_idx} format error: {line}")

            smiles = splitted[0]
            name   = splitted[1] if len(splitted) > 1 else ""

            bond_props   = splitted[2] if len(splitted) > 2 else ""
            amide_bonds  = splitted[3] if len(splitted) > 3 else ""

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Warning: Invalid SMILES on line {line_idx}: {smiles}")
                raise

            mol.SetProp("_Name", name)

            if bond_props:
                if bond_props == '':
                    pass
                else:
                    # '|': [ 'idx:value', 'idx:value', ... ]
                    for prop in bond_props.split('|'):
                        if not prop:
                            continue
                        idx, value = prop.split(':')
                        bond = mol.GetBondWithIdx(int(idx))
                        bond.SetProp("LinkingBond", value)

            if amide_bonds:
                if amide_bonds == '':
                    pass
                else:
                    for prop in amide_bonds.split('|'):
                        if not prop:
                            continue
                        idx, value = prop.split(':')
                        bond = mol.GetBondWithIdx(int(idx))
                        bond.SetProp("AmideBond", value)

            # assert Chem.Mol of mol
            # assert isinstance(mol, Chem.Mol)
            yield mol

if __name__ == "__main__":
    print("Please run script mol2graph.py, do not run this script directly!")
    raise