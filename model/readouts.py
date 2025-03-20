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

        for linear in self.linears:
            nn.init.normal_(linear.weight, mean=0, std=(d_model ** -0.5))
            nn.init.zeros_(linear.bias)
        
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

        for attn_layer in [self.a_attn, self.p_attn, self.rsd_attn]:
            nn.init.normal_(attn_layer.weight, std=0.01)
            nn.init.zeros_(attn_layer.bias)

        nn.init.xavier_uniform_(self.cross_proj.weight)
        nn.init.zeros_(self.cross_proj.bias)

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

    def forward(self, bg, suffix='h'):
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
    
class SemanticAttentionReadout(nn.Module):
    def __init__(self, hid_dim, attn_heads=1):
        super().__init__()
        self.hid_dim = hid_dim
        self.attn_heads = attn_heads

        # 1. 分类型注意力池化（Step 1）
        self.type_attn = nn.ModuleDict({
            'a': nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                nn.Tanh(),
                nn.Linear(hid_dim, 1, bias=False)
            ),
            'p': nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                nn.Tanh(),
                nn.Linear(hid_dim, 1, bias=False)
            ),
            'rsd': nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                nn.Tanh(),
                nn.Linear(hid_dim, 1, bias=False)
            )
        })

        # 2. 类型间注意力（Step 2）
        self.semantic_attn = MultiHeadedAttention(h=attn_heads, d_model=hid_dim)
        self.semantic_proj = nn.Linear(hid_dim, 1, bias=False)

        # 初始化参数
        for attn in self.type_attn.values():
            nn.init.xavier_normal_(attn[0].weight)
            nn.init.xavier_normal_(attn[2].weight)
        nn.init.xavier_normal_(self.semantic_proj.weight)

    def get_batch_indices(self, bg, ntype):
        """生成每个节点对应的图索引（batch内第几个图）"""
        counts = bg.batch_num_nodes(ntype).to(bg.device)
        return torch.repeat_interleave(
            torch.arange(bg.batch_size, device=bg.device),
            counts
        )

    def type_wise_pool(self, bg, ntype):
        """对指定类型的节点做注意力池化"""
        if bg.num_nodes(ntype) == 0:
            # 若该类型无节点，返回全零向量
            return torch.zeros(bg.batch_size, self.hid_dim, device=bg.device)
        
        feats = bg.nodes[ntype].data['f_h']  # [num_nodes, hid_dim]
        batch_indices = self.get_batch_indices(bg, ntype)  # [num_nodes]

        # 计算注意力得分（带非线性）
        attn_scores = self.type_attn[ntype](feats)  # [num_nodes, 1]
        attn_scores = attn_scores.squeeze(-1)       # [num_nodes]

        # 按图做 softmax 并加权求和
        weights = F.softmax(attn_scores, dim=0)    # [num_nodes]
        weighted_feats = feats * weights.unsqueeze(-1)
        pooled = scatter(weighted_feats, batch_indices, dim=0, reduce='sum')
        return pooled  # [batch_size, hid_dim]

    def forward(self, bg):
        a_pool = self.type_wise_pool(bg, 'a')  # [B, D]
        p_pool = self.type_wise_pool(bg, 'p')
        rsd_pool = self.type_wise_pool(bg, 'rsd')
        
        pooled_matrix = torch.stack([a_pool, p_pool, rsd_pool], dim=1)  # [B, 3, D]
        
        # 多头注意力交互
        attn_output = self.semantic_attn(
            pooled_matrix, pooled_matrix, pooled_matrix
        )  # [B, 3, D]
        
        # 计算各类型的权重
        weights = F.softmax(
            self.semantic_proj(attn_output).squeeze(-1),  # [B, 3]
            dim=-1
        )  # [B, 3]
        
        # 加权求和
        graph_emb = (attn_output * weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        return graph_emb

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
        
        self.linear_out = nn.Linear(hid_dim*2, hid_dim)

        for gru in [self.a_gru, self.r_gru]:
            for name, param in gru.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)  # 正交初始化
                elif 'bias' in name:
                    nn.init.zeros_(param)

        nn.init.xavier_uniform_(self.linear_out.weight)
        nn.init.zeros_(self.linear_out.bias)

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
    
    def _cross_attn(self, query_pad, query_mask, key_pad, key_mask, attn_layer):
        cross_mask = query_mask.unsqueeze(-1) | key_mask.unsqueeze(1)
        attended = attn_layer(
            query=query_pad, key=key_pad, value=key_pad, mask=cross_mask
        )
        return query_pad + attended

    def forward(self, bg, suffix='h'):
        device = bg.device

        # feature padding
        a_padded, a_mask, a_sizes = self._batch_pad(bg, 'a', f'f_{suffix}')
        p_padded, p_mask, p_sizes = self._batch_pad(bg, 'p', f'f_{suffix}')
        r_padded, r_mask, r_sizes = self._batch_pad(bg, 'rsd', f'f_{suffix}')

        # cross-attentions
        a_from_p = self._cross_attn(a_padded, a_mask, p_padded, p_mask, self.a_cross_attn)  # [B, La, hid_dim]
        a_from_r = self._cross_attn(a_padded, a_mask, r_padded, r_mask, self.r_cross_attn)  # [B, La, hid_dim]
        a_updated = (a_from_p + a_from_r) * 0.5

        # GRUs
        a_out, _ = self.a_gru(a_updated) # [B, La, hid_dim*2]

        # aggregate
        a_embed = self._pool(a_out, a_mask, a_sizes, device)  # [B, hid_dim*2]

        return self.linear_out(a_embed)
    
class TCNReadout(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 num_layers=3,
                 kernel_size=3,
                 dropout=0.1,
                 causal=True,
                 use_residual=True,
                 use_batchnorm=False):
        super().__init__()
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.causal = causal
        self.use_residual = use_residual
        self.use_batchnorm = use_batchnorm
        
        # 逐层构建 TCN blocks
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        in_channels = input_dim
        for i in range(num_layers):
            dilation = 2 ** i  # 指数级扩张
            # 若 causal=True, 则用右侧padding + 剪裁
            # 若非causal, 一般做 same padding => padding = dilation*(kernel_size//2)
            pad = (kernel_size - 1) * dilation if causal else (kernel_size//2)*dilation
            
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                padding=pad,
                dilation=dilation
            )
            self.convs.append(conv)
            
            if use_batchnorm:
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.bns.append(nn.Identity())
            
            in_channels = hidden_dim  # 下一层输入 = 当前层输出
        # 最终投影层
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, bg: dgl.DGLHeteroGraph, rsd_feat_field='f_h'):
        padded_seq, mask = self._pad_rsd_by_pos(bg, rsd_feat_field)  # [B, L, D]
        if padded_seq.size(1) == 0:
            # 若完全无 rsd 节点，则返回全0向量或其他替代
            bsz = padded_seq.size(0)
            zero_feat = padded_seq.new_zeros(bsz, self.proj.out_features)
            return zero_feat
        
        # [B, L, D] -> [B, D, L]
        x = padded_seq.transpose(1, 2)  # x: [B, D, L]
        # 逐层因果/非因果卷积
        for i in range(self.num_layers):
            conv = self.convs[i]
            bn   = self.bns[i]
            if self.use_residual:
                residual = x  # 残差
            
            out = conv(x)
            
            # 对于 causal TCN，需要去掉右侧多余padding
            if self.causal:
                pad_amount = conv.padding[0]
                if pad_amount > 0:
                    out = out[:, :, :-pad_amount]
            
            out = bn(out)
            out = F.relu(out)
            out = F.dropout(out, p=self.dropout, training=self.training)
            
            if self.use_residual:
                # 若维度匹配，则可以加残差
                if out.size() == residual.size():
                    out = out + residual
            x = out
        # [B, D, L] -> [B, L, D]
        x = x.transpose(1, 2)
        # 池化
        pooled = self._masked_mean_pool(x, mask)  # [B, D]
        # 投影
        graph_feat = self.proj(pooled)
        return graph_feat
    
    def _pad_rsd_by_pos(self, bg, feat_field):
        device = bg.device
        graphs = dgl.unbatch(bg)
        batch_size = len(graphs)
        feats_list = []
        lengths = []
        for g in graphs:
            if g.num_nodes('rsd') == 0:
                # 空图
                feats_list.append(torch.zeros(0, 0, device=device))
                lengths.append(0)
                continue
            pos = g.nodes['rsd'].data['pos']
            sorted_idx = torch.argsort(pos)
            feats = g.nodes['rsd'].data[feat_field][sorted_idx]  # [num_rsd, D]
            feats_list.append(feats)
            lengths.append(feats.size(0))
        max_len = max(lengths) if lengths else 0
        if max_len == 0:
            # 全都空
            return (torch.zeros(batch_size, 0, 0, device=device),
                    torch.ones(batch_size, 0, dtype=torch.bool, device=device))
        
        input_dim = feats_list[0].size(-1) if feats_list else 0
        padded_seq = torch.zeros(batch_size, max_len, input_dim, device=device)
        mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)
        for i, (feats, l) in enumerate(zip(feats_list, lengths)):
            if l > 0:
                padded_seq[i, :l] = feats
                mask[i, :l] = False
        return padded_seq, mask
    
    def _masked_mean_pool(self, seq_out, mask):
        """
        seq_out: [B, L, D]
        mask: [B, L] (True 表示填充无效)
        """
        seq_out = seq_out.masked_fill(mask.unsqueeze(-1), 0.0)
        sums = seq_out.sum(dim=1)  # [B, D]
        lens = (~mask).sum(dim=1, keepdim=True).clamp(min=1)
        return sums / lens.float()