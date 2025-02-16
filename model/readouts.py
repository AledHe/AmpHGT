"""bunch of readout functions for graph classification"""

import torch
import dgl
import dgl.nn.pytorch as dglnn

from torch import nn, scatter
from torch.nn import functional as F

# First, we run the effect of the multi-view GNN itself and make a global mean pooling as a baseline; 
# then we add a Cross-Attn readout that only performs self-attention on (a, p) for comparison.
# be careful with batched graphs, readout require batch splitting.

class GlobalPooling(nn.Module):
    def __init__(self, mode='mean'):
        super().__init__()
        self.mode = mode.lower()
        assert self.mode in ['mean', 'sum', 'max', 'concat']

    def batch_pool(self, bg: dgl.DGLHeteroGraph, ntype, feat_name):
        with bg.local_scope():
            # get batch number of nodes
            batch_num_nodes = bg.batch_num_nodes(ntype)
            
            # handle empty graphs
            if batch_num_nodes.sum() == 0:
                return torch.zeros(bg.batch_size, bg.nodes[ntype].data[feat_name].shape[-1]).to(bg.device)
            
            # global pooling
            if self.mode == 'mean':
                pooled = dgl.mean_nodes(bg, ntype, feat_name)
            elif self.mode == 'sum':
                pooled = dgl.sum_nodes(bg, ntype, feat_name)
            elif self.mode == 'max':
                pooled = dgl.max_nodes(bg, ntype, feat_name)
            
            # handle empty graphs
            mask = (batch_num_nodes == 0)
            pooled[mask] = 0
            return pooled

    def forward(self, bg: dgl.DGLHeteroGraph, suffix='h'):
        # global pooling
        a_embed = self.batch_pool(bg, 'a', f'f_{suffix}')
        p_embed = self.batch_pool(bg, 'p', f'f_{suffix}')
        
        if self.mode == 'concat':
            return torch.cat([a_embed, p_embed], dim=-1)
        elif self.mode == 'max':
            # perform element-level max pooling on a_embed and p_embed again
            # after stack => [2, B, hid_dim], then perform max on dim=0 => [B, hid_dim]
            return torch.max(torch.stack([a_embed, p_embed], dim=0), dim=0)[0]
        else:
            # for 'mean' and 'sum', here is (a + p) / 2, can also change it to a + p
            return (a_embed + p_embed) / 2
    
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

class HeteroConvReadout(nn.Module):
    def __init__(self, 
                 hidden_dim, 
                 num_layers=2,
                 agg_type='mean' # mean, sum, max
                 ):
        super().__init__()
        # 1) Define heterogeneous convolution. For each edge type, give a GraphConv.
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.num_layers = num_layers

        for _ in range(num_layers):
            conv = dglnn.HeteroGraphConv({
                ('a', 'b', 'a'): dglnn.GraphConv(hidden_dim, hidden_dim),
                ('p', 'r', 'p'): dglnn.GraphConv(hidden_dim, hidden_dim),
                ('a', 'j', 'p'): dglnn.GraphConv(hidden_dim, hidden_dim),
                ('p', 'j', 'a'): dglnn.GraphConv(hidden_dim, hidden_dim)
            }, aggregate=agg_type)
            self.convs.append(conv)

            # Layer Normalization
            self.norms.append(nn.ModuleDict({
                'a': nn.LayerNorm(hidden_dim),
                'p': nn.LayerNorm(hidden_dim)
            }))

        self.linear_a = nn.Linear(hidden_dim, hidden_dim)
        self.linear_p = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, bg: dgl.DGLHeteroGraph, suffix='h'):
        for layer_idx in range(self.num_layers):
            h_dict = {'a': bg.nodes['a'].data[f'f_{suffix}'],
                      'p': bg.nodes['p'].data[f'f_{suffix}']}
            
            out_dict = self.convs[layer_idx](bg, h_dict)
            # out_dict contains the updated representation of each node type

            for ntype in out_dict:
                out_dict[ntype] = self.norms[layer_idx][ntype](out_dict[ntype])
                out_dict[ntype] = F.relu(out_dict[ntype])

            # update graph node representation
            bg.nodes['a'].data[f'f_{suffix}'] = out_dict['a']
            bg.nodes['p'].data[f'f_{suffix}'] = out_dict['p']

        with bg.local_scope():
            a_feat = bg.nodes['a'].data[f'f_{suffix}']
            p_feat = bg.nodes['p'].data[f'f_{suffix}']
            a_feat = self.linear_a(a_feat)
            p_feat = self.linear_p(p_feat)
            bg.nodes['a'].data[f'f_{suffix}'] = a_feat
            bg.nodes['p'].data[f'f_{suffix}'] = p_feat

        return bg # for further pooling.