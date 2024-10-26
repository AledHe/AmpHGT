import random
import time
from typing import List, Optional, Set, Tuple
import dgl
import torch

from utils.features import ATOM_FEATURES, factory, allowable_features

from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.BRICS import FindBRICSBonds

from utils.FraSCESS.mol2heterograph_FraSCESS import (maccskeys_emb, pharm_property_types_feats, bond_features, 
                                                     atom_labels, map_frag_atom_to_mol, get_pharm_label, 
                                                     atom_features)

def brics_molecule(mol):
    """
    Fragment a molecule using the BRICS algorithm.
    Args:
        smiles (str): A SMILES string representing the molecule.
    Returns:
        list of str: A list of SMILES strings representing the fragments.
    """
    # Convert SMILES to RDKit molecule object
    if isinstance(object, str):
        mol = Chem.MolFromSmiles(mol)

    try:
        Chem.SanitizeMol(mol)
    except ValueError as e:
        print(f"Error sanitizing molecule: {e}")
        return []
    
    # for every atom in the molecule, mark its idx in the mol as a property for mapping
    for atom in mol.GetAtoms():
        atom.SetProp("orig_idx", str(atom.GetIdx()))
        
    break_bonds = [mol.GetBondBetweenAtoms(i[0][0],i[0][1]).GetIdx() for i in FindBRICSBonds(mol)]

    # get terminal atoms
    atom_bond_atom = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds() if bond.GetIdx() in break_bonds]

    return break_bonds, atom_bond_atom

def get_fragment_feats_other(mol: Chem.Mol, sdf_path: str = None):
    # Get fragments from BRICS
    bonds_to_break_all, break_bonds_atoms = brics_molecule(mol)

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
            emb_0 = maccskeys_emb(mol_pharm) # what? you don't know Molecular ACCess System? well...
                                             # To be brief, it is a fixed-length bit string that represents substructure or fragment of a molecule.
                                             # as [0, 0, 0, 0, 0, 1, ..., 0, 1...]
                                             # however, it cannot encode structural features that are not pre-defined in the fragment library.
                                             # for MACCS, only the 166 fragment definitions are available to the public.
                                             # feature 1.
            emb_0 = emb_0 + [0]
            emb_1 = pharm_property_types_feats(mol_pharm) # generates pharmacophore property features for the fragment molecule mol_pharm.
            emb_1 = emb_1 + [0]
        except Exception:
            emb_0 = [0 for i in range(168)]
            emb_1 = [0 for i in range(28)]

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

def mol2heterograph_BRICS(mol: Chem.Mol, sdf_path: str = None, vocab_dict: dict = None) -> dgl.DGLHeteroGraph:
    edge_types = [('a', 'b', 'a'), ('p', 'r', 'p'), ('a', 'j', 'p'), ('p', 'j', 'a')]
    edges = {k: [] for k in edge_types}

    # a, b, a: atom-to-atom bonds
    # p, r, p: fragment-to-fragment (pharmacophore-to-pharmacophore) reactions
    # a, j, p: atom-to-fragment junctions, connection from an atoms to the fragment they belongs to.
    # p, j, a: fragment-to-atom junctions, connection from a fragment to the atoms they belongs to.

    result_ap, result_p, result_frag, result, brics_bonds_rules, bonds_to_break_all  = get_fragment_feats_other(mol, sdf_path=sdf_path)
    
    atom_frag_label = map_frag_atom_to_mol(mol, bonds_to_break_all)

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
        if get_pharm_label(result_frag[result_ap[int(idx)]], vocab_dict) == 0: # don't mask amide bond atoms.
            atom_canmask.append(False)
        else:
            atom_canmask.append(True)

    # print(f_atom)
    f_atom = torch.FloatTensor(f_atom) # convert the atom features to a tensor.
    g.nodes['a'].data['f'] = f_atom # set the atom features to the graph.
    atom_label_list = torch.LongTensor(atom_label_list) # convert the atom labels to a tensor.
    g.nodes['a'].data['label'] = atom_label_list
    atom_canmask = torch.BoolTensor(atom_canmask)
    g.nodes['a'].data['pep'] = atom_canmask # pep is the amide bond label, if false, atom will not be masked.
                                            # more specifically, the NC=O struct (all atom) won't be mask.
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
        pharm_label_list.append(get_pharm_label(frag, vocab_dict))
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