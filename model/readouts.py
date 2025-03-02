"""bunch of readout functions for graph classification"""
import copy
import math
import torch
import dgl

from torch import nn
import torch.nn.functional as F

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

# nn modules

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    # p_attn = F.softmax(scores, dim = -1).masked_fill(mask, 0)  # 不影响
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

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
        rsd_embed = self.batch_pool(bg, 'rsd', feat_name)  # (B, D)

        return a_embed + p_embed + rsd_embed # (B, D)
    
class HierAttnReadout(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        # initialize attention layers for different node types
        self.a_attn = nn.Linear(hid_dim, 1)      # atom-view attention
        self.p_attn = nn.Linear(hid_dim, 1)      # pharmacophore-view attention
        self.rsd_attn = nn.Linear(hid_dim, 1)    # residue-view attention
        
        # fusion layer combining all views
        self.cross_proj = nn.Linear(3*hid_dim, hid_dim)

    def get_batch_indices(self, bg, ntype):
        """generate batch indices for specific node type"""
        device = bg.device
        batch_counts = bg.batch_num_nodes(ntype).to(device)
        return torch.repeat_interleave(
            torch.arange(bg.batch_size, device=device),
            batch_counts
        )

    def batch_attn_pool(self, feats, attn_weights, batch_ids):
        """attention-based pooling with softmax normalization"""
        # unnormalized weights using exp for stability
        unnorm_weights = attn_weights.exp()  # [total_nodes, 1]
        
        # compute softmax denominator per graph
        sum_weights = scatter(unnorm_weights, batch_ids, dim=0, reduce='sum')  # [batch_size, 1]
        sum_weights = sum_weights[batch_ids]  # expand to node level
        norm_weights = unnorm_weights / (sum_weights + 1e-8)  # [total_nodes, 1]
        
        # apply weighted aggregation
        weighted_feats = feats * norm_weights
        return scatter(weighted_feats, batch_ids, dim=0, reduce='sum')  # [batch_size, hid_dim]

    def forward(self, bg, suffix='final'):
        # define processing order and corresponding attention layers
        node_types = ['a', 'p', 'rsd']
        attn_modules = [self.a_attn, self.p_attn, self.rsd_attn]
        pooled_embeddings = []

        for ntype, attn_fn in zip(node_types, attn_modules):
            # get node features and process attention
            batch_indices = self.get_batch_indices(bg, ntype)
            node_features = bg.nodes[ntype].data[f'f_{suffix}']
            attention_scores = attn_fn(node_features)
            
            # perform attention pooling
            graph_embed = self.batch_attn_pool(
                node_features, 
                attention_scores,
                batch_indices
            )
            pooled_embeddings.append(graph_embed)

        # combine different views and project
        unified_embed = torch.cat(pooled_embeddings, dim=-1)
        return self.cross_proj(unified_embed)

class GRUReadout(nn.Module):
    def __init__(self, hid_dim, bidirectional=True):
        super().__init__()
        self.hid_dim = hid_dim
        self.num_directions = 2 if bidirectional else 1
        
        # cross-attentions
        self.a_cross_attn = MultiHeadedAttention(h=4, d_model=hid_dim)
        self.r_cross_attn = MultiHeadedAttention(h=4, d_model=hid_dim)
        
        # GRUs
        self.a_gru = nn.GRU(hid_dim, hid_dim, batch_first=True, bidirectional=bidirectional)
        self.r_gru = nn.GRU(hid_dim, hid_dim, batch_first=True, bidirectional=bidirectional)
        
        # fusion
        self.fusion = nn.Linear(hid_dim*self.num_directions*2, hid_dim*self.num_directions)

    def _batch_pad(self, bg, ntype, field):
        feat = bg.nodes[ntype].data[field]
        node_sizes = bg.batch_num_nodes(ntype)
        max_len = max(node_sizes.max().item(), 1)
        
        batch_size = bg.batch_size
        padded = feat.new_zeros(batch_size, max_len, feat.size(-1))
        mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=feat.device)
        
        col = torch.arange(max_len, device=feat.device).expand(batch_size, -1)
        valid = col < node_sizes.unsqueeze(-1)
        
        padded[valid] = feat
        mask[valid] = False
        return padded, mask, node_sizes
    
    def _pool(self, out, mask, sizes, device):
        mask = ~mask.unsqueeze(-1)  # reverse mask
        summed = (out * mask).sum(dim=1)
        lengths = sizes.clamp(min=1).unsqueeze(-1)
        return summed / lengths.to(device)
    
    def _gru_init(self, x, gru):
        hidden = x.mean(dim=1)  # [B, D]
        hidden = hidden.repeat(gru.num_layers, 1, 1)  # maybe multiple layers
        return gru(x, hidden)
    
    def _cross_attn(self, query_pad, query_mask, key_pad, key_mask, attn_layer):
        cross_mask = query_mask.unsqueeze(-1) | key_mask.unsqueeze(1)
        attended = attn_layer(
            query=query_pad, key=key_pad, value=key_pad, mask=cross_mask
        )
        return query_pad + attended

    def forward(self, bg, suffix='final'):
        device = bg.device

        # feature padding
        a_padded, a_mask, a_sizes = self._batch_pad(bg, 'a', f'f_{suffix}')
        p_padded, p_mask, p_sizes = self._batch_pad(bg, 'p', f'f_{suffix}')
        r_padded, r_mask, r_sizes = self._batch_pad(bg, 'rsd', f'f_{suffix}')

        # cross-attentions
        a_updated = self._cross_attn(a_padded, a_mask, p_padded, p_mask, self.a_cross_attn)
        r_updated = self._cross_attn(r_padded, r_mask, p_padded, p_mask, self.r_cross_attn)

        # GRUs
        a_out, _ = self._gru_init(a_updated, self.a_gru)
        r_out, _ = self._gru_init(r_updated, self.r_gru)

        # aggregate
        a_embed = self._pool(a_out, a_mask, a_sizes, device)
        r_embed = self._pool(r_out, r_mask, r_sizes, device)

        # fusion
        combined = torch.cat([a_embed, r_embed], dim=-1)
        return self.fusion(combined)