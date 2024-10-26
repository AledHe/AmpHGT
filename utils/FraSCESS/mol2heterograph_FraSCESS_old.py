"""
Break molecule method designed by this research.

Fragmentize Side Chain Even Sub-Sidechain (FraSCESS)

Similar to AdaFrag, but we strictly distinguish between side chains and main chains as well as terminal groups.
"""

import time
import dgl
import torch

from utils.features import ATOM_FEATURES, factory, allowable_features

from rdkit import Chem
from rdkit.Chem import MACCSkeys

def mol2heterograph_frascess(mol: Chem.Mol, sdf_path: str = None) -> dgl.DGLHeteroGraph:
    """Convert a RDKit molecule to a DGL heterograph."""
    edge_types = [('a', 'b', 'a'), ('p', 'r', 'p'), ('a', 'j', 'p'), ('p', 'j', 'a')]
    edges = {k: [] for k in edge_types}

    # a, b, a: atom-to-atom bonds
    # p, r, p: fragment-to-fragment (pharmacophore-to-pharmacophore) reactions
    # a, j, p: atom-to-fragment junctions, connection from an atom to the fragment it belongs to.
    # p, j, a: fragment-to-atom junctions, connection from a fragment to the atom it belongs to.

    result_ap, result_p, result_frag, result, brics_bonds_rules, bonds_to_break  = get_fragment_feats(mol, sdf_path=sdf_path)
    
    atom_atom_label = map_frag_atom_to_mol(mol, bonds_to_break)

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
    f_atom = []
    src,dst = g.edges(etype=('a','b','a'))

    atom_label_list = []
    atom_aa_label_list = []
    atom_ismask = []
    # go through all nodes of type 'a' (atoms) in the graph.
    for idx in g.nodes('a'):
        atom = mol.GetAtomWithIdx(idx.item())
        atom_label_list.append(atom_labels(atom))
        atom_aa_label_list.append(atom_atom_label[idx.item()])
        f_atom.append(atom_features(atom))
        if get_pharm_label(result_frag[result_ap[int(idx)]])==1:
            atom_ismask.append(False)
        else:
            atom_ismask.append(True)

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

def get_pharm_label(frag:str):
    if frag not in vocab_dict.keys():
        print(f'Warning Unfound fragments {frag}')
        pass
    else:
        # print(frag)
        pass
    return vocab_dict.get(frag, len(vocab_dict))

def map_atom_indices(fragment, mol):
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
    bonds_to_break, break_bonds_atoms = get_bond_by_prop(mol)

    if len(bonds_to_break) == 0:
        print("Warning: No bonds to break in the peptide.")
        print(f"This rarely happens, please check the input molecule {sdf_path} if you're not sure.")
        temp_mol = mol
    else:
        # Break the bonds
        temp_mol = Chem.FragmentOnBonds(mol, bonds_to_break, addDummies=False)
    
    # fragment_idx_list contains the indices of the atoms in each fragment.
    fragment_idx_list = Chem.GetMolFrags(temp_mol) # List of fragmentized molecules by index
    # fragment_mol_list contains the individual fragment molecules.
    fragment_mol_list = Chem.GetMolFrags(temp_mol, asMols=True) # List of fragmentized molecules

    print(fragment_idx_list)
    print(fragment_mol_list)

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
    # reverse direction.
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

    return result_ap, result_p, result_frag, result, brics_bonds_rules, bonds_to_break

def get_bond_by_prop(mol: Chem.Mol):
    """
    Break bonds in a molecule by a property
    The bonds to be broken are those that have the property "LinkingBond"
    return bonds idx list and (atom, atom) tuple list.
    """
    # Find the bonds to break
    bonds_to_break = []
    atom_bond_atom = []
    for bond in mol.GetBonds():
        if bond.HasProp("LinkingBond"):
            bonds_to_break.append(bond.GetIdx())
            # for each bond, get the indices of the atoms it connects.
            atom_bond_atom.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

    return bonds_to_break, atom_bond_atom

def get_backbone_cut_idx(mol: Chem.Mol):
    """
    Cut the amino acid chain backbone by breaking the bonds between the alpha carbon and the nitrogen.
    """
    bonds_to_break = []
    atom_bond_pairs = []

    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        
        if (atom1.GetSymbol() == "N" and atom2.GetSymbol() == "C") or (atom1.GetSymbol() == "C" and atom2.GetSymbol() == "N"):
            carbon_atom = atom2 if atom1.GetSymbol() == "N" else atom1
            oxygen_neighbor_found = False
            
            for neighbor in carbon_atom.GetNeighbors():
                if neighbor.GetSymbol() == "O":
                    for neighbor_bond in neighbor.GetBonds():
                        if neighbor_bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                            oxygen_neighbor_found = True
                            break
                if oxygen_neighbor_found:
                    break
            
            if oxygen_neighbor_found:
                bonds_to_break.append(bond.GetIdx())
                atom_bond_pairs.append((atom1.GetIdx(), atom2.GetIdx()))

    if not bonds_to_break:
        print("No valid backbone bonds found in the molecule")

    return bonds_to_break, atom_bond_pairs

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