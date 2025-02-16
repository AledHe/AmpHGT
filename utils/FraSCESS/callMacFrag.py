"""
Break the structure with bond connection retained by MacFrag
by Yanyan Diao https://academic.oup.com/bioinformatics/article/39/1/btad012/6986129
"""

from rdkit import Chem
from MacFrag.MacFrag import searchBonds

def call_MacFrag(frag: Chem.Mol, mol: Chem.Mol, mapping_atom: list[Chem.Bond], all_bonds_to_break: set, sdf_path: str, maxSR: int = 5):
    """
    Call MacFrag to FraSCESS by YC. H.

    params:
    - maxSR: only cyclic bonds in smallest SSSR ring of size larger than this value will be cleaved
    see: https://depth-first.com/articles/2020/08/31/a-smallest-set-of-smallest-rings/
    """
    # Call MacFrag to fragment the sidechain
    # Generate candidate bonds using MacFrag's searchBonds logic
    candidate_bonds = list(searchBonds(frag, maxSR=maxSR))

    for bond_info in candidate_bonds:
        a1_frag, a2_frag = bond_info[0]
        bond_in_frag = frag.GetBondBetweenAtoms(a1_frag, a2_frag)
        if not bond_in_frag:
            continue
        bond_idx_frag = bond_in_frag.GetIdx()
        
        # Get corresponding bond in the original molecule
        a1_mol = mapping_atom[a1_frag]
        a2_mol = mapping_atom[a2_frag]
        original_bond = mol.GetBondBetweenAtoms(a1_mol, a2_mol)
        if not original_bond:
            continue
        bond_idx_ori = original_bond.GetIdx()
        
        # Check if adjacent to already broken bonds
        begin_atom_ori = original_bond.GetBeginAtom().GetIdx()
        end_atom_ori = original_bond.GetEndAtom().GetIdx()
        adjacent_bonds = set()
        for bond in mol.GetAtomWithIdx(begin_atom_ori).GetBonds():
            adjacent_bonds.add(bond.GetIdx())
        for bond in mol.GetAtomWithIdx(end_atom_ori).GetBonds():
            adjacent_bonds.add(bond.GetIdx())
        adjacent_bonds.discard(bond_idx_ori)
        
        if adjacent_bonds & all_bonds_to_break:
            continue  # Adjacent to existing breaks, skip
        
        # Check if breaking this bond splits the fragment
        temp_frag = Chem.FragmentOnBonds(frag, [bond_idx_frag], addDummies=False)
        try:
            frags = Chem.GetMolFrags(temp_frag, asMols=True)
        except Chem.rdchem.KekulizeException as e:
            print(f"KekulizeException when fragmenting bond {bond_idx_ori} in {sdf_path}")
            continue
        
        if len(frags) > 1:
            all_bonds_to_break.add(bond_idx_ori)
            return 0 # Exit after successfully breaking one bond
    
    # If no valid bonds found, return without changes
    return 1