"""
Break molecule method designed by this research.

Fragmentize Side Chain Even Sub-Sidechain (FraSCESS)

Similar to AdaFrag, but we strictly distinguish between side chains and main chains as well as terminal groups.
"""

import random
import time
from typing import List, Optional, Set, Tuple
import dgl
import torch

from utils.features import ATOM_FEATURES, factory, allowable_features

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

    result_ap, result_p, result_frag, result, brics_bonds_rules, bonds_to_break_all  = get_fragment_feats(mol, sdf_path=sdf_path)
    
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

def get_pharm_label(frag:str, vocab_dict:dict):
    if frag not in vocab_dict.keys():
        print(f'Warning Unfound fragments {frag}')
        pass
    else:
        # print(frag)
        pass
    return vocab_dict.get(frag, len(vocab_dict) - 1)

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

    return bonds_to_break, atom_bond_pairs

def fragmentize_by_bond(mol: Chem.Mol, bonds_to_break: List[int], sdf_path: str, FraSCESS: bool) -> Tuple[List[int], List[Chem.Mol]]:
    # for every atom in the molecule, mark its idx in the mol as a property for mapping
    for atom in mol.GetAtoms():
        atom.SetProp("orig_idx", str(atom.GetIdx()))

    all_bonds_to_break = set(bonds_to_break)

    if not bonds_to_break:
        print(f"Warning: No bonds to break in the peptide. Please check the input molecule {sdf_path}.")
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
                    if FraSCESS:
                        # Randomly break bonds until fragment is divided into at least two substructures
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

def fragmentize_sidechain(frag: Chem.Mol, mol: Chem.Mol, all_bonds_to_break: set, sdf_path: str, seed: int = 42):
    # print(Chem.MolToSmiles(frag))
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
    Return the fragment index with the most N-C=O substructure.
    """
    bond_smiles = "C(=O)N"
    max_match = 0
    idx = None
    
    for i, frag in enumerate(frags_mol_lst):
        matches = frag.GetSubstructMatches(Chem.MolFromSmarts(bond_smiles))
        if len(matches) > max_match:
            max_match = len(matches)
            idx = i

    if idx is None:
        print(f"Cannot find backbone. Please check inputs {sdf_path}")
    
    return idx

def maccskeys_emb(mol):
    return list(MACCSkeys.GenMACCSKeys(mol))

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