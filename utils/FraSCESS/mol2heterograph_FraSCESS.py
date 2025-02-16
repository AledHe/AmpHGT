"""
Break molecule method designed by this research.

Fragmentize Side Chain Even Sub-Sidechain (FraSCESS)

Similar to AdaFrag, but we strictly distinguish between side chains and main chains as well as terminal groups.
"""

import functools
import random
import dgl
import torch

from typing import List, Optional, Tuple

from utils.features import ATOM_FEATURES, factory, allowable_features
from utils.std_logger import warning
from .callMacFrag import call_MacFrag

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.BRICS import FindBRICSBonds

def mol2heterograph_frascess(mol: Chem.Mol, sdf_path: str = None, vocab_dict: dict = None) -> dgl.DGLHeteroGraph:
    """
    Convert a RDKit molecule to a DGL heterograph.
    Most code in this function is from PepLand
    """
    edge_types = [('a', 'b', 'a'), ('p', 'r', 'p'), ('a', 'j', 'p'), ('p', 'j', 'a')]
    edges = {k: [] for k in edge_types}

    # a, b, a: atom-to-atom bonds
    # p, r, p: fragment-to-fragment (pharmacophore-to-pharmacophore) reactions
    # a, j, p: atom-to-fragment junctions, connection from an atoms to the fragment they belongs to.
    # p, j, a: fragment-to-atom junctions, connection from a fragment to the atoms they belongs to.

    result_ap, result_p, result_frag, result, brics_bonds_rules, bonds_to_break_all = get_fragment_feats_o(mol, sdf_path=sdf_path)

    # result_ap_, result_p_, result_frag_, result_, brics_bonds_rules_, bonds_to_break_all_ = get_fragment_feats_o(mol, sdf_path=sdf_path)

    # print(result_ap == result_ap_, result_p == result_p_, result_frag == result_frag_, result == result_, brics_bonds_rules == brics_bonds_rules_, bonds_to_break_all == bonds_to_break_all_)

    # exit()
    
    # atom_frag_label = map_frag_atom_to_mol(mol, bonds_to_break_all)
    atom_frag_label = map_frag_atom_to_mol_o(mol, bonds_to_break_all)

    # assert atom_frag_label == atom_frag_label_

    # print("reac_idx", result) [[0, 0, 1], [14, 16, 15],...
    # print("bbr", brics_bonds_rules) [[[0, 1], [0, True, False, False...
    # print(result_frag) {0: 'CO', 1: 'NCC=O', 2: 'Cc1c[nH]c2ccccc12',...

    # creating bi-directional edges
    # alpha. atom view.
    for bond in mol.GetBonds(): 
        edges[('a','b','a')].append([bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()])
        edges[('a','b','a')].append([bond.GetEndAtomIdx(),bond.GetBeginAtomIdx()])

    # beta. fragment view.
    for r in result:
        begin = r[1]
        end = r[2]
        edges[('p','r','p')].append([result_ap[begin],result_ap[end]])
        edges[('p','r','p')].append([result_ap[end],result_ap[begin]])

    # gamma. junction view.
    for k,v in result_ap.items():
        edges[('a','j','p')].append([k,v])
        edges[('p','j','a')].append([v,k])

    g = dgl.heterograph(edges)
    # Atom Features
    f_atom = []
    src,dst = g.edges(etype=('a','b','a'))

    # print(src)
    # print(dst)

    atom_label_list = []
    atom_aa_label_list = []
    atom_canmask = []
    # go through all nodes of type 'a' (atoms) in the graph.
    for idx in g.nodes('a'):
        # print(idx) # tensor(0)
        atom = mol.GetAtomWithIdx(idx.item())
        atom_label_list.append(atom_labels(atom)) # get and append the atom features.
        # print(atom_label_list, atom_labels(atom)) # [[5, 0], ...] [5, 0] [atomic number, chirality]
        atom_aa_label_list.append(atom_frag_label[idx.item()]) # append the atom 
        # print(atom_frag_label) {0: 0, 1: 0, ..., 17: 5, 18: 5, 19: 5, ...,  111: 29, 112: 29, 114: 29} {atomidx: fragmentidx}
        # print(atom_aa_label_list) [0, 0, 0, 0, 1, 1, 1, 2, 3, 3, 3...] each atom and its corresponding fragment index.
        f_atom.append(atom_features(atom))
        # print(f_atom, atom_features(atom))
        # print(result_frag[result_ap[int(idx)]])
        if get_pharm_label(result_frag[result_ap[int(idx)]], vocab_dict, sdf_path) == 0: # don't mask alpha backbone atoms.
            atom_canmask.append(False)
        else:
            atom_canmask.append(True)

    # print(atom_canmask)

    # print(f_atom)
    f_atom = torch.FloatTensor(f_atom) # convert the atom features to a tensor.
    g.nodes['a'].data['f'] = f_atom # set the atom features to the graph.
    atom_label_list = torch.LongTensor(atom_label_list) # convert the atom labels to a tensor.
    g.nodes['a'].data['label'] = atom_label_list
    atom_canmask = torch.BoolTensor(atom_canmask)
    g.nodes['a'].data['pep'] = atom_canmask # pep is the alpha backbone label, if false, atom will not be masked.
                                            # more specifically, the NCC=O struct (all atom) won't be mask.
    atom_aa_label_list = torch.LongTensor(atom_aa_label_list)
    g.nodes['a'].data['aa_label'] = atom_aa_label_list # atom and its corresponding fragment index.

    '''
    print(Chem.MolToSmiles(mol))
    print(g.number_of_nodes('a'))
    print(g.nodes['a'].data['f'], f"Shape: {g.nodes['a'].data['f'].size()}")
    print(g.nodes['a'].data['label'], f"Shape: {g.nodes['a'].data['label'].size()}")
    print(g.nodes['a'].data['pep'], f"Shape: {g.nodes['a'].data['pep'].size()}")
    print(g.nodes['a'].data['aa_label'], f"Shape: {g.nodes['a'].data['aa_label'].size()}")
    exit()
    '''
    
    dim_atom = len(f_atom[0])

    # Fragment Features
    f_pharm = []
    pharm_label_list = []
    # print(result_p) {0: [0, 0,..], 1: [[0, 1,...]...} pharmacophore features for each fragment.
    for k,v in result_p.items():
        frag = result_frag[k] # get the SMILES of the fragment molecule.
        pharm_label_list.append(get_pharm_label(frag, vocab_dict, sdf_path))
        f_pharm.append(v)

    # print(f_pharm) # concatenated pharmacophore features for each fragment.
    # print(pharm_label_list) # corresponding pharmacophore label for each fragment, label from vocab_dict

    g.nodes['p'].data['f'] = torch.FloatTensor(f_pharm)
    dim_pharm = len(f_pharm[0])
    pharm_label_list = torch.LongTensor(pharm_label_list)
    g.nodes['p'].data['label'] = pharm_label_list

    # Padding Features
    dim_atom_padding = g.nodes['a'].data['f'].size()[0]
    dim_pharm_padding = g.nodes['p'].data['f'].size()[0]
    # concatenates the atom features and the zero tensor along the feature dimension
    g.nodes['a'].data['f_junc'] = torch.cat([g.nodes['a'].data['f'], torch.zeros(dim_atom_padding, dim_pharm)], 1)
    # concatenates the zero tensor and the pharmacophore features along the feature dimension
    g.nodes['p'].data['f_junc'] = torch.cat([torch.zeros(dim_pharm_padding, dim_atom), g.nodes['p'].data['f']], 1)

    # atom and pharmacophore nodes end up with features of the same dimension (dim_atom + dim_pharm), 238 in this case.
    # the new feature f_junc for both atom and pharmacophore nodes includes both their original features and padding, 
    # for the model to handle them in a unified manner.

    # Atom-to-Atom Edge Features
    f_bond = []
    src,dst = g.edges(etype=('a','b','a'))
    for i in range(g.num_edges(etype=('a','b','a'))):
        f_bond.append(bond_features(mol.GetBondBetweenAtoms(src[i].item(),dst[i].item())))
    g.edges[('a','b','a')].data['x'] = torch.FloatTensor(f_bond)

    # Fragment-to-Fragment Edge Features
    f_reac = []
    # result_ap: Dict {atom_id:pharm_id}
    src, dst = g.edges(etype=('p','r','p'))

    for idx in range(g.num_edges(etype=('p','r','p'))):
        p0_g = src[idx].item()
        p1_g = dst[idx].item()
        
        for i in brics_bonds_rules:
            p0 = result_ap[i[0][0]]
            p1 = result_ap[i[0][1]]
            if p0_g == p0 and p1_g == p1:
                f_reac.append(i[1])
                # this means pharmacophore A has more than 1 bonds with pharmacophore B (multi-connection)
                break

    g.edges[('p','r','p')].data['x'] = torch.FloatTensor(f_reac)

    # print dim of atom, fragment, and bond features
    # print(f"Atom Feature Dim: {dim_atom}, Fragment Feature Dim: {dim_pharm}, Bond Feature Dim: {len(f_bond[0])}")

    return g

def atom_labels(atom):
    atom_feature = [allowable_features['possible_atomic_num_list'].
                    index(atom.GetAtomicNum())] + [allowable_features['possible_chirality_list'].
                    index(atom.GetChiralTag())]
    return atom_feature

def atom_features(atom: Chem.rdchem.Atom):
    features = onek_encoding_unk(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
           onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
           [1 if atom.GetIsAromatic() else 0] + \
           [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    return features

def get_pharm_label(frag:str, vocab_dict:dict, sdf_path: str):
    if frag not in vocab_dict.keys():
        print(f'Warning Unfound fragments {frag} in {sdf_path}') # suppress the warning
    else:
        # print(frag)
        pass
    return vocab_dict.get(frag, len(vocab_dict) - 1)

def map_frag_atom_to_mol_o(mol: Chem.Mol, bonds_to_break: list):
    """
    optimized version of map_frag_atom_to_mol, fall back to it if you think the code is too slow.
    """
    for atom in mol.GetAtoms():
        atom.SetProp('orig_idx', str(atom.GetIdx()))
    
    if bonds_to_break:
        fragmented_mol = Chem.FragmentOnBonds(mol, bonds_to_break, addDummies=False)
        mol_fragments = Chem.GetMolFrags(fragmented_mol, asMols=True)
        aa_label = {}
        for frag_idx, frag in enumerate(mol_fragments):
            for atom in frag.GetAtoms():
                orig_idx = int(atom.GetProp('orig_idx'))
                aa_label[orig_idx] = frag_idx
        return aa_label
    else:
        # no bonds to break, return the original molecule
        return {atom.GetIdx(): 1 for atom in mol.GetAtoms()}

def map_atom_indices(fragment, mol):
    """
    To retrieve the original atom indices from the fragment.
    """
    return [mol.GetAtomWithIdx(int(atom.GetProp('orig_idx'))).GetIdx() for atom in fragment.GetAtoms()]

def map_frag_atom_to_mol(mol: Chem.Mol, bonds_to_break: list):
    """
    Map the atoms in the fragments to the original molecule.
    Code from Pepland
    """
    for atom in mol.GetAtoms():
        atom.SetProp('orig_idx', str(atom.GetIdx()))

    if bonds_to_break:
        mol_fragments = Chem.GetMolFrags(Chem.FragmentOnBonds(mol, bonds_to_break, addDummies=False), asMols=True)
    else:
        # fail to break the molecule, return the original molecule
        aa_label = {}
        atom_indices_map = map_atom_indices(mol, mol)
        for atom_idx in atom_indices_map:
            aa_label[atom_idx] = 1
        return aa_label
    
    aa_idx = 0
    aa_label = {}
    for frag in mol_fragments:
            # map the atoms in the fragment to the original molecule
            atom_indices_map = map_atom_indices(frag, mol)
            for atom_idx in atom_indices_map:
                aa_label[atom_idx] = aa_idx
            aa_idx += 1

    return aa_label

### get_fragment_feats

def get_fragment_feats_o(mol: Chem.Mol, sdf_path: str = None):
    bonds_to_break = get_bond_by_prop(mol)
    bonds_to_break_all, break_bonds_atoms = fragmentize_by_bond(mol, bonds_to_break, sdf_path, FraSCESS=True)
    temp_mol = Chem.FragmentOnBonds(mol, bonds_to_break_all, addDummies=False)
    fragment_idx_list = Chem.GetMolFrags(temp_mol)
    fragment_mol_list = Chem.GetMolFrags(temp_mol, asMols=True)

    result_ap, result_p, result_frag = {}, {}, {}
    pharm_id = 0

    for pharm_id, fragment_idx in enumerate(fragment_idx_list):
        for atom_id in fragment_idx:
            result_ap[atom_id] = pharm_id
        mol_pharm = fragment_mol_list[pharm_id]
        try:
            emb_0 = maccskeys_emb_o(mol_pharm) + [0]
            emb_1 = pharm_property_types_feats_o(mol_pharm) + [0]
        except Exception:
            emb_0 = [0 for i in range(168)]
            emb_1 = [0 for i in range(28)]
            warning(f"Failed to generate features for fragment {pharm_id} in {sdf_path}")
        result_p[pharm_id] = emb_0 + emb_1
        result_frag[pharm_id] = Chem.MolToSmiles(mol_pharm, canonical=True)

    brics_bonds_set = set()
    brics_bonds_rules = []
    for item in break_bonds_atoms:
        a, b = int(item[0]), int(item[1])
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is None:
            continue
        feats = bond_features(bond)
        brics_bonds_set.update({(a, b), (b, a)})
        brics_bonds_rules.extend([[[a, b], feats], [[b, a], feats]])

    result = []
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if (begin, end) in brics_bonds_set:
            result.append([bond.GetIdx(), begin, end])

    return result_ap, result_p, result_frag, result, brics_bonds_rules, bonds_to_break_all

def get_fragment_feats(mol: Chem.Mol, sdf_path: str = None):
    # get bond to break
    bonds_to_break = get_bond_by_prop(mol)

    bonds_to_break_all, break_bonds_atoms = fragmentize_by_bond(mol, bonds_to_break, sdf_path, FraSCESS=True)

    temp_mol = Chem.FragmentOnBonds(mol, bonds_to_break_all, addDummies=False)

    # fragment_idx_list contains the indices of the atoms in each fragment.
    fragment_idx_list = Chem.GetMolFrags(temp_mol) # List of fragmentized molecules by index
    # fragment_mol_list contains the individual fragment molecules.
    fragment_mol_list = Chem.GetMolFrags(temp_mol, asMols=True) # List of fragmentized molecules

    # print(fragment_idx_list) ((0, 5), (1, 2, 3, 4),...
    # print(fragment_mol_list) (<rdkit.Chem.rdc...

    result_ap = {} # atom-pharmacophore mapping
    result_p = {}
    result_frag = {}
    pharm_id = 0

    # for each fragment, mapping atoms to their corresponding pharmacophore ID (I call it fragments id, pharmacophore is because the codes are from PharmHGT).
    for fragment_idx in fragment_idx_list:
        for atom_id in fragment_idx:
            result_ap[atom_id] = pharm_id # maps atoms to their corresponding pharmacophore ID.
        try: # after mapping atoms to their corresponding pharmacophore ID, we generate the pharmacophore features for each fragment.
            mol_pharm = Chem.MolFromSmiles(Chem.MolFragmentToSmiles(mol, fragment_idx))
            emb_0 = maccskeys_emb_o(mol_pharm) # In case you don't know Molecular ACCess System
                                             # To be brief, it is a fixed-length bit string that represents substructure or fragment of a molecule.
                                             # as [0, 0, 0, 0, 0, 1, ..., 0, 1...]
                                             # however, it cannot encode structural features that are not pre-defined in the fragment library.
                                             # for MACCS, only the 166 fragment definitions are available to the public.
                                             # feature 1.
            # emb_0 = maccskeys_emb(mol_pharm)
            # emb_0_o = maccskeys_emb_o(mol_pharm)

            # assert emb_0 == emb_0_o

            emb_0 = emb_0 + [0]
            emb_1 = pharm_property_types_feats_o(mol_pharm) # generates pharmacophore property features for the fragment molecule mol_pharm.
            # emb_1 = pharm_property_types_feats(mol_pharm)
            # emb_1_o = pharm_property_types_feats_o(mol_pharm)
            
            # assert emb_1 == emb_1_o

            emb_1 = emb_1 + [0]
        except Exception:
            emb_0 = [0 for i in range(168)]
            emb_1 = [0 for i in range(28)]
            warning(f"Failed to generate MACCS keys for fragment {Chem.MolToSmiles(fragment_mol_list[pharm_id])} in {sdf_path}")

        result_p[pharm_id] = emb_0 + emb_1 # result/sum the pharmacophore features for each fragment.
        result_frag[pharm_id] = Chem.MolToSmiles(fragment_mol_list[pharm_id], canonical=True) # get the SMILES of the fragment molecule.
        pharm_id += 1 # next fragment in the loop.

    # Generate the bonds feature between fragments.
    # bidirection.
    brics_bonds = list()
    brics_bonds_rules = list()
    for item in break_bonds_atoms:
        brics_bonds.append([int(item[0]), int(item[1])])
        # calculates features for the bond between the two atoms in breaked atoms.
        brics_bonds_rules.append([[int(item[0]), int(item[1])], bond_features(mol.GetBondBetweenAtoms(int(item[0]), int(item[1])))])
        brics_bonds.append([int(item[1]), int(item[0])])
        brics_bonds_rules.append([[int(item[1]), int(item[0])], bond_features(mol.GetBondBetweenAtoms(int(item[0]), int(item[1])))])

    result = []
    # get all bonds in the molecule
    for bond in mol.GetBonds():
        beginatom = bond.GetBeginAtomIdx()
        endatom = bond.GetEndAtomIdx()
        # if any bond is in the brics_bonds list, append it to the result list.
        # so it is fragment-bond-fragment bond.
        if [beginatom, endatom] in brics_bonds:
            result.append([bond.GetIdx(), beginatom, endatom])
        # finally, the result holds the bonds that connects fragments by holding begin and end atoms.

    return result_ap, result_p, result_frag, result, brics_bonds_rules, bonds_to_break_all

### get_fragment_feats

def get_bond_by_prop(mol: Chem.Mol) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Break bonds in a molecule by a property 'LinkingBond'.
    """
    bonds_to_break = [bond.GetIdx() for bond in mol.GetBonds() if bond.HasProp("LinkingBond")]
    # atom_bond_atom = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds() if bond.HasProp("LinkingBond")]
    return bonds_to_break

def get_backbone_cut_idx(mol: Chem.Mol, sdf_path: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    Identify the bonds between the alpha carbon and the nitrogen in the amino acid backbone.
    """
    bonds_to_break = []
    atom_bond_pairs = []

    for bond in mol.GetBonds():
        # if bonds have prop "AmideBonds", then break
        if bond.HasProp("AmideBond"):
            bonds_to_break.append(bond.GetIdx())
            atom_bond_pairs.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

    # if no bonds have prop "AmideBonds", then break the bond between the alpha carbon and the nitrogen
    if not bonds_to_break:
        print("Please set 'AmideBond' prop to your SMILES, or it may lead inaccurate results.")
        for bond in mol.GetBonds():
            atom1, atom2 = bond.GetBeginAtom(), bond.GetEndAtom()
            if {atom1.GetSymbol(), atom2.GetSymbol()} == {"N", "C"}:
                carbon_atom = atom2 if atom1.GetSymbol() == "N" else atom1
                if any(neighbor.GetSymbol() == "O" and 
                    any(nb.GetBondType() == Chem.rdchem.BondType.DOUBLE for nb in neighbor.GetBonds())
                    for neighbor in carbon_atom.GetNeighbors()):
                    bonds_to_break.append(bond.GetIdx())
                    atom_bond_pairs.append((atom1.GetIdx(), atom2.GetIdx()))

    if not bonds_to_break:
        print(f"No valid backbone bonds found in the molecule, {Chem.MolToSmiles(mol)}, {sdf_path}")
        print(f"This should not happen. But code will continue...")

    return bonds_to_break, atom_bond_pairs

def fragmentize_by_bond(mol: Chem.Mol, bonds_to_break: List[int], sdf_path: str, FraSCESS: bool) -> Tuple[List[int], List[Chem.Mol]]:
    # for every atom in the molecule, mark its idx in the mol as a property for mapping
    for atom in mol.GetAtoms():
        atom.SetProp("orig_idx", str(atom.GetIdx()))

    all_bonds_to_break = set(bonds_to_break)

    if not bonds_to_break:
        print(f"Warning: No sidechain bonds to break in the peptide. Please check the input molecule {Chem.MolToSmiles(mol)}.")
        print(f"But we will try to proceed...")
        # Directly process by backbone_cut
        backbone_idx = get_backbone([mol], sdf_path)
        bonds_to_break_bk, _ = get_backbone_cut_idx(mol, sdf_path)
        if bonds_to_break_bk:
            mapping_atom = map_atom_indices(mol, mol)
            for bond_idx in bonds_to_break_bk:
                bond = mol.GetBondWithIdx(bond_idx)
                begin_atom_idx = mapping_atom[bond.GetBeginAtomIdx()]
                end_atom_idx = mapping_atom[bond.GetEndAtomIdx()]
                original_bond = mol.GetBondBetweenAtoms(begin_atom_idx, end_atom_idx)
                original_bond_idx = original_bond.GetIdx()
                all_bonds_to_break.add(original_bond_idx)
    else:
        # Break the bonds
        temp_mol = Chem.FragmentOnBonds(mol, bonds_to_break, addDummies=False)
        temp_mol_tuple = Chem.GetMolFrags(temp_mol, asMols=True)

        backbone_idx = get_backbone(temp_mol_tuple, sdf_path)
        
        for i, frag in enumerate(temp_mol_tuple):
            if i != backbone_idx:
                # set prop of the all atoms in the fragment
                for atom in frag.GetAtoms():
                    atom.SetProp("sidechain", str(i))
                brics_bonds = [frag.GetBondBetweenAtoms(i[0][0],i[0][1]).GetIdx() for i in FindBRICSBonds(frag)]
                if brics_bonds:
                    mapping_atom = map_atom_indices(frag, mol)
                    # print(brics_bonds)
                    for bond_idx in brics_bonds:
                        bond = frag.GetBondWithIdx(bond_idx)
                        begin_atom_idx = mapping_atom[bond.GetBeginAtomIdx()]
                        end_atom_idx = mapping_atom[bond.GetEndAtomIdx()]
                        original_bond = mol.GetBondBetweenAtoms(begin_atom_idx, end_atom_idx)
                        original_bond_idx = original_bond.GetIdx()
                        all_bonds_to_break.add(original_bond_idx)
                else:
                    # print SMILES of the fragment
                    # print(Chem.MolToSmiles(frag))
                    if FraSCESS:
                        fragmentize_sidechain(frag, mol, all_bonds_to_break, sdf_path)

        for j, frag in enumerate(temp_mol_tuple):
            if j == backbone_idx:
                # set prop of the all atoms in the fragment
                for atom in frag.GetAtoms():
                    atom.SetProp("backbone", str(j))
                bonds_to_break_bk, _ = get_backbone_cut_idx(frag, sdf_path)
                if bonds_to_break_bk:
                    mapping_atom = map_atom_indices(frag, mol)
                    for bond_idx in bonds_to_break_bk:
                        bond = frag.GetBondWithIdx(bond_idx)
                        begin_atom_idx = mapping_atom[bond.GetBeginAtomIdx()]
                        end_atom_idx = mapping_atom[bond.GetEndAtomIdx()]
                        original_bond = mol.GetBondBetweenAtoms(begin_atom_idx, end_atom_idx)
                        original_bond_idx = original_bond.GetIdx()
                        all_bonds_to_break.add(original_bond_idx)

    # get atom-bond-atom idx for each bond in all_bonds_to_break
    atom_bond_atom = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds() if bond.GetIdx() in all_bonds_to_break]

    return all_bonds_to_break, atom_bond_atom

def fragmentize_sidechain(frag: Chem.Mol, mol: Chem.Mol, all_bonds_to_break: set, sdf_path: str):
    mapping_atom = map_atom_indices(frag, mol)

    return_flag = call_MacFrag(frag, mol, mapping_atom, all_bonds_to_break, sdf_path)

    if return_flag == 0: # success
        return
    elif return_flag == 1: # cannot fragmentize further
        # print(f"Warning: failed to fragmentize a sidechain. {sdf_path}, {Chem.MolToSmiles(frag)}") # too noisy
        return

def fragmentize_sidechain_dep(frag: Chem.Mol, mol: Chem.Mol, all_bonds_to_break: set, sdf_path: str, seed: int = 42):
    # print(Chem.MolToSmiles(frag))
    # TODO: Implement MacFrag instead of Random.
    # We will need to modify MacFrag to adapt our style.
    # if a sidechain is not fragmentizable in any way, mark it for not masking.
    random.seed(seed)
    bonds = []
    for bond in frag.GetBonds():
        if bond.GetBondType() == Chem.BondType.SINGLE:
            bonds.append(bond.GetIdx())

    if not bonds:
        return

    mapping_atom = map_atom_indices(frag, mol)

    while bonds:
        bond_idx_frag = random.choice(bonds)
        bond_in_frag = frag.GetBondWithIdx(bond_idx_frag)
        begin_atom_idx_frag = bond_in_frag.GetBeginAtomIdx()
        end_atom_idx_frag = bond_in_frag.GetEndAtomIdx()

        original_bond = mol.GetBondBetweenAtoms(mapping_atom[begin_atom_idx_frag], mapping_atom[end_atom_idx_frag])

        # to make sure that the bond is not neighboring any bonds in all_bonds_to_break
        # get the terminal atoms of the bond in the original molecule
        begin_atom_idx_ori = original_bond.GetBeginAtom().GetIdx()
        end_atom_idx_ori = original_bond.GetEndAtom().GetIdx()
        # get the bonds of the terminal atoms
        begin_atom_bonds = [bond.GetIdx() for bond in mol.GetAtomWithIdx(begin_atom_idx_ori).GetBonds()]
        end_atom_bonds = [bond.GetIdx() for bond in mol.GetAtomWithIdx(end_atom_idx_ori).GetBonds()]
        # check if any of the terminal atoms' bonds are in all_bonds_to_break
        if any(bond_idx in all_bonds_to_break for bond_idx in begin_atom_bonds) or any(bond_idx in all_bonds_to_break for bond_idx in end_atom_bonds):
            bonds.remove(bond_idx_frag)
            # restart the loop
            continue

        # get original bond idx
        bond_idx_ori = original_bond.GetIdx()
        
        # Ensure the selected bond does not reconnect the initial split atoms
        if bond_idx_ori not in all_bonds_to_break:
            all_bonds_to_break.add(bond_idx_ori)
            
            temp_frag = Chem.FragmentOnBonds(frag, [bond_idx_frag], addDummies=False)
            try:
                frag_tuple = Chem.GetMolFrags(temp_frag, asMols=True)
            except Chem.rdchem.KekulizeException as e:
                print("########## DEBUG ##########")
                print(f"Error: {e}")
                print(f"Fragment: {Chem.MolToSmiles(frag)}")
                print(f"Molecule: {Chem.MolToSmiles(mol)}, {sdf_path}")
                # print each atom and its idx
                for atom in frag.GetAtoms():
                    print(atom.GetSymbol(), atom.GetIdx())
                # print each bond and terminal atoms, bond type, bond idx
                for bond in frag.GetBonds():
                    print(bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol(), bond.GetBondType(), bond.GetIdx())
                print(f"Bond to break: {bond_idx_frag}, terminal atom symbols: {frag.GetAtomWithIdx(begin_atom_idx_frag).GetSymbol()}, {frag.GetAtomWithIdx(end_atom_idx_frag).GetSymbol()}")
                print(f"Bond to break type: {bond_in_frag.GetBondType()}")
                print("########## DEBUG ##########")
                raise e
            # print(frag_tuple)
            
            if len(frag_tuple) > 1:
                break

            bonds.remove(bond_idx_frag)
        else:
            # print(f"Warning: Bond {bond_idx} is in all_bonds_to_break.")
            bonds.remove(bond_idx_frag)

def get_backbone(frags_mol_lst: List[Chem.Mol], sdf_path: str) -> Optional[int]:
    """
    Return the fragment index fit the rule.
    """
    bond_smiles = "[CX3](=O)[NX3;H1]"
    max_match = 0
    max_prop_count = 0
    idx = None
    
    for i, frag in enumerate(frags_mol_lst):
        # The backbone fragment is the fragment have the most N-C=O substructure and 
        # have the most bond with AmideBond prop.
        match_count = len(frag.GetSubstructMatches(Chem.MolFromSmarts(bond_smiles)))
        
        prop_count = sum(1 for bond in frag.GetBonds() if bond.HasProp("AmideBond"))

        if prop_count > max_prop_count or (match_count > max_match and prop_count >= max_prop_count):
            max_match = match_count
            max_prop_count = prop_count
            # print(f"Match count: {match_count}, Prop count: {prop_count}, idx: {i}")
            idx = i

    if idx is None:
        print(f"Cannot find backbone. Please check inputs {sdf_path}")
    
    return idx

### maccskeys_emb

@functools.lru_cache()
def _maccskeys_emb_cached(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    return list(MACCSkeys.GenMACCSKeys(mol))

def get_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

def maccskeys_emb_o(mol: Chem.Mol):
    smiles = get_smiles(mol)
    return _maccskeys_emb_cached(smiles)

def maccskeys_emb(mol):
    return list(MACCSkeys.GenMACCSKeys(mol))

### maccskeys_emb

### pharm_property_types_feats

@functools.lru_cache()
def _pharm_property_types_feats_cached(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    types = [i.split('.')[1] for i in factory.GetFeatureDefs().keys()]
    feats = [i.GetType() for i in factory.GetFeaturesForMol(mol)]
    result = [0] * len(types)
    for i in range(len(types)):
        if types[i] in list(set(feats)):
            result[i] = 1
    return result

def pharm_property_types_feats_o(mol):
    smiles = get_smiles(mol)
    return _pharm_property_types_feats_cached(smiles)

def pharm_property_types_feats(mol,factory=factory): 
    # extracts the types of pharmacophore features from the feature factory definitions.
    types = [i.split('.')[1] for i in factory.GetFeatureDefs().keys()]
    # gets the actual pharmacophore features present in the molecule.
    feats = [i.GetType() for i in factory.GetFeaturesForMol(mol)]
    result = [0] * len(types)
    # creates a binary vector where each position corresponds to a specific pharmacophore type.
    for i in range(len(types)):
        if types[i] in list(set(feats)):
            result[i] = 1
    return result

### pharm_property_types_feats

def bond_features(bond: Chem.rdchem.Bond):
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))

    return fbond

def onek_encoding_unk(value, choices):
    # initialize a zero vector of length len(choices) + 1
    encoding = [0] * (len(choices) + 1)
    # find the index of the value in choices
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding