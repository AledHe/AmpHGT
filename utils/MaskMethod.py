import random
import torch
import dgl
from utils.std_logger import info
from typing import Dict, List, Optional, Tuple

from .features import ATOM_FEATURES
from .FraSCESS.mol2heterograph_FraSCESS import onek_encoding_unk

class MaskGraph:
    def __init__(self, 
                 num_atom_type: int, 
                 num_edge_type: int, 
                 mask_edge: bool = True,
                 mask_atom: bool = True, 
                 mask_fragment: bool = True, 
                 mask_edge_r: float = 0.1,
                 mask_atom_r: float = 0.1, 
                 mask_fragment_r: float = 0.1,
                 ):
        """
        According to the input parameters, mask the graph by following the rules:
        (1)	Random masking: Atoms not part of the backbone chain of the peptide graph are randomly masked.
        (2)	Fragment masking: Randomly masking fragments in the peptide graph, which can include sidechain, parts of the sidechain or sections of the backbone.
        
        :param num_atom_type: Number of atom types in the graph.
        :param num_edge_type: Number of edge types in the graph.
        :param mask_edge: Whether to mask edges in the graph.
        :param mask_atom: Whether to mask atoms in the graph.
        :param mask_fragment: Whether to mask fragments in the graph.
        :param mask_atom_r: The ratio of atoms to be masked.
        :param mask_fragment_r: The ratio of fragments to be masked.

        A large portion of this code is adapted from Pepland.
        """
        # Validate input parameters
        self._validate_init_params(num_atom_type, num_edge_type, 
                                 mask_edge_r, mask_atom_r, mask_fragment_r)

        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_edge = mask_edge
        self.mask_atom = mask_atom
        self.mask_fragment = mask_fragment
        self.mask_edge_r = mask_edge_r
        self.mask_atom_r = mask_atom_r
        self.mask_fragment_r = mask_fragment_r
        
        # Cache feature dimensions for validation
        self.atom_feat_dim = self._get_atom_feat_dim()
        self.fragment_feat_dim = self._get_fragment_feat_dim()

        info(f"Initialized MaskGraph with atom_feat_dim={self.atom_feat_dim}, "
                    f"fragment_feat_dim={self.fragment_feat_dim}")
        
    def _validate_init_params(self, 
                              num_atom_type: int, 
                              num_edge_type: int, 
                              mask_edge_r: float, 
                              mask_atom_r: float, 
                              mask_fragment_r: float
                              ):
        """Validate initialization parameters."""
        if not all(isinstance(x, int) and x > 0 for x in [num_atom_type, num_edge_type]):
            raise ValueError("num_atom_type and num_edge_type must be positive integers")
        
        if not all(isinstance(x, float) and 0 <= x <= 1 for x in [mask_edge_r, mask_atom_r, mask_fragment_r]):
            raise ValueError("Masking ratios must be float values between 0 and 1")
        

    def _get_atom_feat_dim(self) -> int:
        """Get the dimension of atom features."""
        return len(atom_mask_features())
    
    def _get_fragment_feat_dim(self) -> int:
        """Get the dimension of fragment features."""
        return len(GetMaskFragmentFeats())
    
    def _validate_feature_dims(self, g: dgl.DGLHeteroGraph):
        """Validate feature dimensions in the graph."""
        if g.nodes['a'].data['f'].shape[-1] != self.atom_feat_dim:
            raise ValueError(f"Atom feature dimension mismatch. Expected {self.atom_feat_dim}, "
                           f"got {g.nodes['a'].data['f'].shape[-1]}")
        
        if g.nodes['p'].data['f'].shape[-1] != self.fragment_feat_dim:
            raise ValueError(f"Fragment feature dimension mismatch. Expected {self.fragment_feat_dim}, "
                           f"got {g.nodes['p'].data['f'].shape[-1]}")
        
    def _create_mask_indices(self, num_elements: int, mask_ratio: float, 
                           exclude_indices: Optional[List[int]] = None) -> List[int]:
        """Create mask indices with exclusion support."""
        available_indices = list(range(num_elements))
        if exclude_indices:
            available_indices = list(set(available_indices) - set(exclude_indices))
            
        mask_size = max(1, int(len(available_indices) * mask_ratio))
        return random.sample(available_indices, min(mask_size, len(available_indices)))
        
    def __call__(self, g: dgl.DGLHeteroGraph):
        # Validate feature dimensions
        self._validate_feature_dims(g)

        # Store original features for validation
        orig_atom_feats = g.nodes['a'].data['f'].clone()
        orig_frag_feats = g.nodes['p'].data['f'].clone()

        # Fragment masking
        if self.mask_fragment:
            num_pharms = g.number_of_nodes('p')
            masked_pharm_indices = self._create_mask_indices(num_pharms, self.mask_fragment_r)
            
            # Create and store mask
            node_mask = torch.zeros(num_pharms, dtype=torch.bool)
            node_mask[masked_pharm_indices] = True
            g.nodes['p'].data['mask'] = node_mask
            
            # Apply masking
            mask_feats = torch.FloatTensor(GetMaskFragmentFeats())
            g.nodes['p'].data['f'][node_mask] = mask_feats

        # Atom masking with structural awareness
        if self.mask_atom:
            num_atoms = g.number_of_nodes('a')
            # Use structure-independent masking criteria
            maskable_atoms = torch.ones(num_atoms, dtype=torch.bool)
            masked_atom_indices = self._create_mask_indices(num_atoms, self.mask_atom_r)
            
            # Create and store mask
            node_mask = torch.zeros(num_atoms, dtype=torch.bool)
            node_mask[masked_atom_indices] = True
            g.nodes['a'].data['mask'] = node_mask
            
            # Apply masking
            mask_feats = torch.FloatTensor(atom_mask_features())
            g.nodes['a'].data['f'][node_mask] = mask_feats

        # Edge masking (if needed)
        if self.mask_edge:
            self._apply_edge_masking(g)

        # Validate masking results
        self._validate_masking_results(g, orig_atom_feats, orig_frag_feats)

        # exit("Test")
        return g

    def _apply_edge_masking(self, g: dgl.DGLHeteroGraph):
        """Apply edge masking with proper feature handling."""
        num_edges = g.number_of_edges('b')
        masked_edge_indices = self._create_mask_indices(num_edges, self.mask_edge_r)
        
        edge_mask = torch.zeros(num_edges, dtype=torch.bool)
        edge_mask[masked_edge_indices] = True
        g.edges['b'].data['mask'] = edge_mask
        
        # Apply edge feature masking
        if edge_mask.any():
            g.edges['b'].data['x'][edge_mask] = torch.FloatTensor(bond_mask_features())

    def _validate_masking_results(self, g: dgl.DGLHeteroGraph, 
                                orig_atom_feats: torch.Tensor,
                                orig_frag_feats: torch.Tensor):
        """Validate that masking was applied correctly."""
        # Verify that unmasked features remain unchanged
        atom_mask = g.nodes['a'].data.get('mask', torch.zeros(g.number_of_nodes('a'), dtype=torch.bool))
        frag_mask = g.nodes['p'].data.get('mask', torch.zeros(g.number_of_nodes('p'), dtype=torch.bool))
        
        assert torch.allclose(g.nodes['a'].data['f'][~atom_mask], orig_atom_feats[~atom_mask])
        assert torch.allclose(g.nodes['p'].data['f'][~frag_mask], orig_frag_feats[~frag_mask])
            
def GetMaskFragmentFeats():
    emb_0 = [0 for i in range(167)]+[1]
    emb_1 = [0 for i in range(27)]+[1]

    # result_p[pharm_id] = emb_0 + emb_1

    return emb_0+emb_1

def atom_mask_features():
    features = onek_encoding_unk(10, ATOM_FEATURES['atomic_num']) + \
           onek_encoding_unk(10, ATOM_FEATURES['degree']) + \
           onek_encoding_unk(10, ATOM_FEATURES['formal_charge']) + \
           onek_encoding_unk(10, ATOM_FEATURES['chiral_tag']) + \
           onek_encoding_unk(10, ATOM_FEATURES['num_Hs']) + \
           onek_encoding_unk(10, ATOM_FEATURES['hybridization']) + \
           [0] + \
           [0.01]  # scaled to about the same range as other features
    return features

def bond_mask_features():

    fbond = [
        0,  # bond is not None
        0,
        0,
        0,
        0,
        0,
        0
    ]
    fbond += onek_encoding_unk(6, list(range(6)))

    return fbond

def validate_feature_vectors():
    fragment_feature = GetMaskFragmentFeats()
    atom_feature = atom_mask_features()

    print(fragment_feature)
    print(atom_feature)

    assert len(fragment_feature) == 196, f"Unexpected fragment feature length: {len(fragment_feature)}"
    assert len(atom_feature) == 42, f"Unexpected atom feature length: {len(atom_feature)}"

    print("Feature vector validation passed.")

if __name__ == "__main__":
    validate_feature_vectors()