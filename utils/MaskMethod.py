import random
import torch
import dgl
from model.sequence_embedding import TOKEN2ID
from utils.std_logger import info

from .features import ATOM_FEATURES
from .FraSCESS.mol2heterograph_FraSCESS import onek_encoding_unk

# revised the mask graph code, corrected the not masking amide bond to not masking alpha backbone. 25.2.8

class MaskGraph:
    def __init__(self, 
                 num_atom_type: int, 
                 num_edge_type: int, 
                 mask_edge: bool = True,
                 mask_atom: bool = True, 
                 mask_fragment: bool = True, 
                 mask_rsd: bool = True,
                 mask_rate: float = 0.8, # unified mask rate for atom, edge and fragment.
                 # mask_amino: float = 0.0, # mask atoms according to aa_label, this is disabled in pepland.
                 ):
        """
        According to the input parameters, mask the graph by following the rules:
        (1)	Random masking: Atoms not part of the backbone chain of the peptide graph are randomly masked.
        (2)	Fragment masking: Randomly masking fragments in the peptide graph, which can include sidechain, parts of the sidechain or sections of the backbone.
        
        :param num_atom_type: 119 by default, tell model it is a masked atom. the 119 atom is currently not found by human.
        :param num_edge_type: 5 by default, tell model it is a masked bond.
        Note that above two param is not currently used for we've stored labels in g.nodes['a'].data['label'],
        Directly classification features using CrossEntropyLoss to match.
        :param mask_edge: Whether to mask edges in the graph.
        :param mask_atom: Whether to mask atoms in the graph.
        :param mask_fragment: Whether to mask fragments in the graph.
        :param mask_rate: The rate of masking for atom, edge and fragment.
        :param mask_amino: The rate of masking for atom according to aa_label. deprecated

        A large portion of this code is adapted from Pepland.
        Randomness here is okay as it is masking processing.
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_edge = mask_edge
        self.mask_atom = mask_atom
        self.mask_fragment = mask_fragment
        self.mask_rsd = mask_rsd
        self.mask_rate = mask_rate
        # self.mask_amino = mask_amino
        
        # Cache feature dimensions for validation
        self.atom_feat_dim = self._get_atom_feat_dim()
        self.fragment_feat_dim = self._get_fragment_feat_dim()

        info(f"Initialized MaskGraph with atom_feat_dim={self.atom_feat_dim}, "
                    f"fragment_feat_dim={self.fragment_feat_dim}")

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
        
    def __call__(self, g: dgl.DGLHeteroGraph):
        # Validate feature dimensions
        self._validate_feature_dims(g)

        # Store original features for validation
        orig_atom_feats = g.nodes['a'].data['f'].clone()
        orig_frag_feats = g.nodes['p'].data['f'].clone()

        num_atoms = g.number_of_nodes('a')
        final_atom_mask = torch.zeros(num_atoms, dtype=torch.bool)

        # mask atoms according to aa_label, in other words, mask atoms in fragments (for each fragment, masking num_amino * self.mask_amino + 1).
        # if self.mask_atom and self.mask_amino > 0:
        #     amino = torch.unique(g.nodes['a'].data['aa_label'])
        #     print(amino)
        #     num_amino = len(amino)
        #     sample_size = int(num_amino * self.mask_amino + 1)
        #     masked_amino_indices = random.sample(range(num_amino), sample_size)
        #     mask_amino = amino[masked_amino_indices]

        #     atom_mask = torch.zeros(g.num_nodes('a'), dtype=torch.bool)
        #     print(atom_mask)
        #     for aa in mask_amino:
        #         atom_mask |= (g.nodes['a'].data['aa_label'] == aa)

        #     print(atom_mask)

        #     g.nodes['a'].data['f'][atom_mask] = torch.FloatTensor(atom_mask_features()).repeat(atom_mask.sum(), 1)
        #     # g.nodes['a'].data['mask'] = atom_mask
        #     final_atom_mask |= atom_mask

        # mask atoms except for amide bonds (essentially mask again after masked amino atoms, this time for all atoms excerpt for amide bonds)
        if self.mask_atom and self.mask_rate > 0:
            # get maskable, g.nodes['a'].data['pep'] is bool list. when false, the atom cannot be masked.
            maskable_atoms = g.nodes('a')[g.nodes['a'].data['pep']]
            # print(maskable_atoms)
            num_maskable = len(maskable_atoms)
            # print(num_maskable)

            # create mask indices
            sample_size = int(num_maskable * self.mask_rate + 1)

            # pick actual node IDs, not sublist indices
            masked_atom_indices = random.sample(maskable_atoms.tolist(), sample_size)
            
            # masked_atom_indices = random.sample(range(num_maskable), sample_size)
            # print(masked_atom_indices)
            
            node_mask = torch.zeros(num_atoms, dtype=torch.bool)
            # print(node_mask, node_mask.shape) # all false
            node_mask[masked_atom_indices] = True
            # print(node_mask, node_mask.shape) # indices are true
            # g.nodes['a'].data['mask'] = node_mask
            g.nodes['a'].data['f'][node_mask] = torch.FloatTensor(atom_mask_features()).repeat(node_mask.sum(), 1)
            # print(g.nodes['a'].data['f'][node_mask], g.nodes['a'].data['f'][node_mask].shape)
            # print(g.nodes['a'].data['f'], g.nodes['a'].data['f'].shape)
            # also store the unmasked features for target reconstruction
            g.nodes['a'].data['f_unmasked'] = orig_atom_feats
            # print(g.nodes['a'].data['f_unmasked'], g.nodes['a'].data['f_unmasked'].shape)

            # assert there should be different between the masked and unmasked features
            assert not torch.allclose(g.nodes['a'].data['f'][node_mask], g.nodes['a'].data['f_unmasked'][node_mask])

            # print(node_mask)

            final_atom_mask |= node_mask
            
            if self.mask_edge:
                self._apply_edge_masking(g, masked_atom_indices)
        
        # atom masking is done
        g.nodes['a'].data['mask'] = final_atom_mask

        # Fragment masking
        if self.mask_fragment:
            num_pharms = g.number_of_nodes('p')

            sample_size = int(num_pharms * self.mask_rate + 1)
            masked_pharm_indices = random.sample(range(num_pharms), sample_size)

            # print(masked_pharm_indices)
            
            # Create and store mask
            node_mask = torch.zeros(num_pharms, dtype=torch.bool)
            node_mask[masked_pharm_indices] = True
            g.nodes['p'].data['mask'] = node_mask
            
            # Apply masking
            mask_feats = torch.FloatTensor(GetMaskFragmentFeats())
            g.nodes['p'].data['f'][node_mask] = mask_feats

            # also store the unmasked features for target reconstruction
            g.nodes['p'].data['f_unmasked'] = orig_frag_feats

            assert not torch.allclose(g.nodes['p'].data['f'][node_mask], g.nodes['p'].data['f_unmasked'][node_mask])

        # masking rsd
        if self.mask_rsd and self.mask_rate > 0:
            self._mask_rsd_nodes(g)

        # exit()
        # Validate masking results
        # self._validate_masking_results(g, orig_atom_feats, orig_frag_feats)

        # exit("Test")
        return g

    def _mask_rsd_nodes(self, g: dgl.DGLHeteroGraph):
        """mask the token_id in rsd nodes, replace with with <mask> ID."""
        rsd_nodes = g.nodes('rsd')
        num_rsd = len(rsd_nodes)

        sample_size = int(num_rsd * self.mask_rate + 1)
        masked_indices = random.sample(range(num_rsd), sample_size)

        # store the original label
        orig_labels = g.nodes['rsd'].data['label'].clone()
        g.nodes['rsd'].data['label_orig'] = orig_labels

        # apply masking
        mask_token_id = TOKEN2ID.get("<MASK>")
        g.nodes['rsd'].data['label'][masked_indices] = mask_token_id

        mask_tensor = torch.zeros(num_rsd, dtype=torch.bool)
        mask_tensor[masked_indices] = True
        g.nodes['rsd'].data['mask'] = mask_tensor

    def _apply_edge_masking(self, g: dgl.DGLHeteroGraph, masked_atom_indices: torch.Tensor):
        """Apply edge masking with proper feature handling."""
        connected_edge_indices = []
        
        # Get connected edges with masked atoms
        for atom_idx in masked_atom_indices:
            in_edges = g.in_edges(atom_idx, etype='b')
            out_edges = g.out_edges(atom_idx, etype='b')
            connected_edge_indices.extend(in_edges[2].tolist())
            connected_edge_indices.extend(out_edges[2].tolist())
        
        # deduplicate
        connected_edge_indices = list(set(connected_edge_indices))
        
        edge_mask = torch.zeros(g.num_edges('b'), dtype=torch.bool)
        edge_mask[connected_edge_indices] = True
        g.edges['b'].data['mask'] = edge_mask
        orig_bond_feats = g.edges['b'].data['x'].clone()
        g.edges['b'].data['x'][edge_mask] = torch.FloatTensor(bond_mask_features())

        # also store the unmasked features for target reconstruction
        g.edges['b'].data['x_unmasked'] = orig_bond_feats

    def _validate_masking_results(self, g: dgl.DGLHeteroGraph, 
                                orig_atom_feats: torch.Tensor,
                                orig_frag_feats: torch.Tensor):
        """Validate that masking was applied correctly."""
        # Verify that unmasked features remain unchanged
        atom_mask = g.nodes['a'].data.get('mask', torch.zeros(g.number_of_nodes('a'), dtype=torch.bool))
        frag_mask = g.nodes['p'].data.get('mask', torch.zeros(g.number_of_nodes('p'), dtype=torch.bool))
        residue_mask = g.nodes['rsd'].data.get('mask', torch.zeros(g.number_of_nodes('rsd'), dtype=torch.bool))
        
        assert torch.allclose(g.nodes['a'].data['f'][~atom_mask], orig_atom_feats[~atom_mask])
        assert torch.allclose(g.nodes['p'].data['f'][~frag_mask], orig_frag_feats[~frag_mask])
        assert torch.allclose(g.nodes['rsd'].data['label'][~residue_mask], g.nodes['rsd'].data['label_orig'][~residue_mask])
            
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