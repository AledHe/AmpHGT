"""bunch of readout functions for graph classification"""

import torch
import dgl
import dgl.nn.pytorch as dglnn

from torch import nn

from utils.std_logger import error, warning

try:
    from torch_scatter import scatter
except ImportError:
    warning("Please install torch-scatter to enable readout functions.")

# First, we run the effect of the multi-view GNN itself and make a global mean pooling as a baseline; 
# then we add a Cross-Attn readout that only performs self-attention on (a, p) for comparison.
# be careful with batched graphs, readout require batch splitting.

class Permute(nn.Module):
    """ Utility to permute a B x N x D tensor into B x D x N. """
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.permute(x, (0, 2, 1))

class Squeeze(nn.Module):
    """ Utility to squeeze a given dimension. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return torch.squeeze(x, dim=self.dim)
    
def split_batch(bg: dgl.DGLHeteroGraph, ntype: str, field: str, device=None):
    """
    1) Gather node embeddings for a type ntype.
    2) Split by each subgraph in bg (bg.batch_size many) into sub-tensors.
    3) Zero-pad them to the same dimension (max # of nodes among subgraphs).
    4) Return shape: (B, max_num_nodes, hidden_dim).
    """
    hidden = bg.nodes[ntype].data[field]   # shape (N, d_model)
    if device is None:
        device = hidden.device

    node_size = bg.batch_num_nodes(ntype)  # shape (B,)
    start_index = torch.cat([
        torch.tensor([0], device=device),
        torch.cumsum(node_size, dim=0)[:-1]
    ])

    max_num_node = int(node_size.max())
    hidden_lst = []
    for i in range(bg.batch_size):
        start = start_index[i]
        size = node_size[i]
        # gather sub-tensor
        cur_hidden = hidden.narrow(0, start, size)
        # zero-pad to (max_num_node, d_model)
        pad_len = max_num_node - size
        if pad_len > 0:
            pad = torch.zeros(pad_len, cur_hidden.size(1), device=device)
            cur_hidden = torch.cat([cur_hidden, pad], dim=0)
        # shape: (max_num_node, d_model)
        hidden_lst.append(cur_hidden.unsqueeze(0))

    # final shape: (B, max_num_node, d_model)
    hidden_cat = torch.cat(hidden_lst, dim=0)
    return hidden_cat

class GlobalPooling(nn.Module):
    def __init__(self, mode='mean'):
        super().__init__()
        self.mode = mode.lower()
        assert self.mode in ['mean', 'sum', 'max']
    
    def batch_pool(self, bg: dgl.DGLHeteroGraph, ntype: str, feat_name: str) -> torch.Tensor:
        """
        Pool the node features of type ntype in a batched graph bg.
        Returns a tensor of shape (batch_size, hidden_dim).
        """
        # node_feats shape: (Total # nodes of type ntype in batch, hidden_dim)
        if self.mode == 'mean':
            pooled = dgl.mean_nodes(bg, feat_name, ntype=ntype)
        elif self.mode == 'sum':
            pooled = dgl.sum_nodes(bg, feat_name, ntype=ntype)
        elif self.mode == 'max':
            pooled = dgl.max_nodes(bg, feat_name, ntype=ntype)
        else:
            error(f'Invalid pooling mode: {self.mode}')
        return pooled
    
    def forward(self, bg: dgl.DGLHeteroGraph, suffix='h'):
        feat_name = f'f_{suffix}'

        a_embed = self.batch_pool(bg, 'a', feat_name)  # (B, D)
        p_embed = self.batch_pool(bg, 'p', feat_name)  # (B, D)

        return a_embed + p_embed # (B, D)
    
class HierAttnReadout(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        # atom-view attention parameters
        self.a_attn = nn.Linear(hid_dim, 1)
        # pharmacophore-view attention parameters
        self.p_attn = nn.Linear(hid_dim, 1)
        # cross-view fusion
        self.cross_proj = nn.Linear(2*hid_dim, hid_dim)

    def batch_attn_pool(self, feats, attn_weights, batch_indices):
        # feats: [total_nodes, hid_dim]
        # attn_weights: [total_nodes, 1]
        # batch_indices: [total_nodes] reprensenting batch indices for each node
        
        # calculate weighted sum by graph group
        w_unorm = attn_weights.exp()  # shape=[N_total,1]
        # segment sum over batch_indices
        w_sum = scatter(w_unorm, batch_indices, dim=0, reduce='sum')  # [B,1]

        # expand to node level
        w_sum_expanded = w_sum[batch_indices]  # [N_total,1]
        norm_w = w_unorm / (w_sum_expanded + 1e-8)

        weighted = feats * norm_w
        return scatter(weighted, batch_indices, dim=0, reduce='sum')  # [B, hid_dim]

    def forward(self, bg, suffix='h'):
        device = bg.device

        # 1) ============= Atom-view =============
        # get the batch_indices assigned to node a
        a_batch_sizes = bg.batch_num_nodes('a').to(device)
        a_batch_indices = torch.repeat_interleave(
            torch.arange(bg.batch_size, device=device),
            a_batch_sizes
        )
        a_feats = bg.nodes['a'].data[f'f_{suffix}']  # [N_a_total, hid_dim]
        a_scores = self.a_attn(a_feats)              # [N_a_total, 1]
        a_embed  = self.batch_attn_pool(a_feats, a_scores, a_batch_indices)  # [B, hid_dim]

        # 2) ============= Pharmacophore-view =============
        p_batch_sizes = bg.batch_num_nodes('p').to(device)
        p_batch_indices = torch.repeat_interleave(
            torch.arange(bg.batch_size, device=device),
            p_batch_sizes
        )
        p_feats = bg.nodes['p'].data[f'f_{suffix}']  # [N_p_total, hid_dim]
        p_scores = self.p_attn(p_feats)              # [N_p_total, 1]
        p_embed  = self.batch_attn_pool(p_feats, p_scores, p_batch_indices)  # [B, hid_dim]

        # 3) ============= Cross-view fusion =============
        combined = torch.cat([a_embed, p_embed], dim=-1)  # [B, 2*hid_dim]
        return self.cross_proj(combined)                  # [B, hid_dim]