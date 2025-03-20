import os
import json
import tqdm
import multiprocessing
from rdkit import Chem
from functools import partial
from utils.FraSCESS.mol2heterograph_FraSCESS import get_fragment_feats
from utils.RegularMethod.BRICSFrag import get_fragment_feats_other
from utils.data_rework import FinetuneDataLoader

Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.BondProps)

def record_bond_properties(mol):
    bond_properties = {}
    for bond in mol.GetBonds():
        if bond.HasProp("LinkingBond"):
            bond_properties[bond.GetIdx()] = bond.GetProp("LinkingBond")
    return bond_properties

def reassign_bond_properties(mol, bond_properties):
    for bond_idx, prop_value in bond_properties.items():
        mol.GetBondWithIdx(bond_idx).SetProp("LinkingBond", prop_value)

def process_molecule(mol_data, method):
    sdf_path, mol = mol_data
    # reassign_bond_properties(mol, bond_properties)
    if method == "FraSCESS":
        result_ap, result_p, result_frag, result, brics_bonds_rules, bonds_to_break = get_fragment_feats(mol, sdf_path=sdf_path)
    elif method == "BRICS":
        result_ap, result_p, result_frag, result, brics_bonds_rules, bonds_to_break = get_fragment_feats_other(mol, sdf_path=sdf_path)
    else:
        raise ValueError(f"Invalid method: {method}")
    return result_frag

def main(cfg, count_threshold: int, output_path: str, method: str, workers: int):
    # Initialize DataLoader
    vocab_loader = FinetuneDataLoader(cfg, stage="generate_vocabdict", mask_graph=None, sdf_path=cfg.data.dataset)

    # Record bond properties outside multiprocessing
    # bond_properties_dict = {}
    # for idx, mol in enumerate(vocab_loader.mols):
    #     bond_properties_dict[idx] = record_bond_properties(mol)
    
    # Prepare data for multiprocessing
    mol_data_list = [(idx, mol) for idx, mol in enumerate(vocab_loader.mols)]
    
    # Use Manager for shared data structures
    with multiprocessing.Manager() as manager:
        fragment_count = manager.dict()
        total_fragments = manager.list()

        partial_process = partial(process_molecule, method=method)

        # Process molecules in parallel
        with multiprocessing.Pool(processes=workers) as pool:
            for result_frag in tqdm.tqdm(pool.imap(partial_process, mol_data_list), total=len(mol_data_list)):
                for frag in result_frag.values():
                    fragment_count[frag] = fragment_count.get(frag, 0) + 1
                    if frag not in total_fragments:
                        total_fragments.append(frag)

        # Calculate frequent fragments (appearing more than 2 times)
        frequent_fragments = {frag: i for i, (frag, count) in enumerate(fragment_count.items()) if count >= count_threshold}
        unique_frequent_fragments = set(frequent_fragments.keys())
        unique_all_fragments = set(total_fragments)

        # Calculate percentage of frequent fragments
        frequent_fragments_percentage = (len(unique_frequent_fragments) / len(unique_all_fragments)) * 100 if len(unique_all_fragments) > 0 else 0

        # rearrange the index of frequent fragments
        frequent_fragments = {frag: i for i, (frag, _) in enumerate(frequent_fragments.items())}

        # Prepare output data
        output_data = {
            "frequent_fragments": frequent_fragments,
            "frequent_fragments_percentage": frequent_fragments_percentage
        }

        # Save the output data to disk
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4)

        print(f"Vocabulary list and percentage saved to {output_path}.")
