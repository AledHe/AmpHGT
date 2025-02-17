import dgl
import torch
from torch import nn
import torch.nn.functional as F
from dgl import function as fn
from functools import partial
import copy

import math
from model.utils import get_func
from utils.std_logger import info
from .readouts import GlobalPooling, HierAttnReadout

# dgl graph utils
def reverse_edge(tensor):
    n = tensor.size(0)
    assert n%2 ==0
    delta = torch.ones(n).type(torch.long)
    delta[torch.arange(1,n,2)] = -1
    return tensor[delta+torch.tensor(range(n))]

def del_reverse_message(edge,field):
    """for g.apply_edges"""
    return {'m': edge.src[field]-edge.data['rev_h']}

def add_attn(node,field,attn):
        feat = node.data[field].unsqueeze(1)
        return {field: (attn(feat,node.mailbox['m'],node.mailbox['m'])+feat).squeeze(1)}

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

class Node_GRU(nn.Module):
    """GRU for graph readout. Implemented with dgl graph"""
    def __init__(self,hid_dim,bidirectional=True):
        super(Node_GRU,self).__init__()
        self.hid_dim = hid_dim
        if bidirectional:
            self.direction = 2
        else:
            self.direction = 1
        self.att_mix = MultiHeadedAttention(6,hid_dim)
        self.gru  = nn.GRU(hid_dim, hid_dim, batch_first=True, 
                           bidirectional=bidirectional)
    
    def split_batch(self, bg, ntype, field, device):
        hidden = bg.nodes[ntype].data[field]
        node_size = bg.batch_num_nodes(ntype)
        start_index = torch.cat([torch.tensor([0],device=device),torch.cumsum(node_size,0)[:-1]])
        max_num_node = max(node_size)
        # padding
        hidden_lst = []
        for i  in range(bg.batch_size):
            start, size = start_index[i],node_size[i]
            assert size != 0, size
            cur_hidden = hidden.narrow(0, start, size)
            cur_hidden = torch.nn.ZeroPad2d((0,0,0,max_num_node-cur_hidden.shape[0]))(cur_hidden)
            hidden_lst.append(cur_hidden.unsqueeze(0))

        hidden_lst = torch.cat(hidden_lst, 0)

        return hidden_lst
        
    def forward(self,bg,suffix='h'):
        """
        bg: dgl.Graph (batch)
        hidden states of nodes are supposed to be in field 'h'.
        """
        self.suffix = suffix
        device = bg.device
        
        p_pharmj = self.split_batch(bg,'p',f'f_{suffix}',device)
        a_pharmj = self.split_batch(bg,'a',f'f_{suffix}',device)

        mask = (a_pharmj!=0).type(torch.float32).matmul((p_pharmj.transpose(-1,-2)!=0).type(torch.float32))==0
        h = self.att_mix(a_pharmj, p_pharmj, p_pharmj,mask) + a_pharmj

        hidden = h.max(1)[0].unsqueeze(0).repeat(self.direction,1,1)
        h, hidden = self.gru(h, hidden)
        
        # unpadding and reduce (mean) h: batch * L * hid_dim
        graph_embed = []
        node_size = bg.batch_num_nodes('p')
        start_index = torch.cat([torch.tensor([0],device=device),torch.cumsum(node_size,0)[:-1]])
        for i  in range(bg.batch_size):
            start, size = start_index[i],node_size[i]
            graph_embed.append(h[i, :size].view(-1, self.direction*self.hid_dim).mean(0).unsqueeze(0))
        graph_embed = torch.cat(graph_embed, 0)

        return graph_embed

def apply_custom_copy_src(g, etype, src_field, out_field, reduce_func):
    g.send_and_recv(
        g.edges(etype=etype),
        message_func=partial(copy_src, src_field=src_field, out_field=out_field),
        reduce_func=reduce_func,
        etype=etype
    )

def copy_src(edges, src_field, out_field):
    return {out_field: edges.src[src_field]}

 
class MVMP(nn.Module):
    def __init__(self,msg_func=add_attn,hid_dim=300,depth=3,view='aba',suffix='h',act=nn.ReLU()):
        """
        MultiViewMassagePassing
        view: a, ap, apj
        suffix: filed to save the nodes' hidden state in dgl.graph. 
                e.g. bg.nodes[ntype].data['f'+'_junc'(in ajp view)+suffix]
        """
        super(MVMP,self).__init__()
        self.view = view
        self.depth = depth
        self.suffix = suffix
        self.msg_func = msg_func
        self.act = act
        self.homo_etypes = [('a','b','a')]
        self.hetero_etypes = []
        self.node_types = ['a','p']
        if 'p' in view:
            self.homo_etypes.append(('p','r','p'))
        if 'j' in view:
            self.node_types.append('junc')
            self.hetero_etypes=[('a','j','p'),('p','j','a')] # don't have feature

        self.attn = nn.ModuleDict()
        for etype in self.homo_etypes + self.hetero_etypes:
            self.attn[''.join(etype)] = MultiHeadedAttention(4,hid_dim)

        self.mp_list = nn.ModuleDict()
        for edge_type in self.homo_etypes:
            self.mp_list[''.join(edge_type)] = nn.ModuleList([nn.Linear(hid_dim,hid_dim) for i in range(depth-1)])

        self.node_last_layer = nn.ModuleDict()
        for ntype in self.node_types:
            self.node_last_layer[ntype] = nn.Linear(3*hid_dim,hid_dim)

    def update_edge(self,edge,layer):
        return {'h':self.act(edge.data['x']+layer(edge.data['m']))}
    
    def update_node(self,node,field,layer):
        return {field:layer(torch.cat([node.mailbox['mail'].sum(dim=1),
                                       node.data[field],
                                       node.data['f']],1))}
    def init_node(self,node):
        return {f'f_{self.suffix}':node.data['f'].clone()}

    def init_edge(self,edge):
        return {'h':edge.data['x'].clone()}


    def forward(self,bg):
        suffix = self.suffix
        if  not bg.edges[('p','r','p')].data['x'].numel() and len(self.homo_etypes)>1:
            self.homo_etypes.remove(('p','r','p'))


        for ntype in self.node_types:
            if ntype != 'junc':
                bg.apply_nodes(self.init_node,ntype=ntype)
                
        for etype in self.homo_etypes:
            bg.apply_edges(self.init_edge,etype=etype)

        
        if 'j' in self.view:
            bg.nodes['a'].data[f'f_junc_{suffix}'] = bg.nodes['a'].data['f_junc'].clone()
            bg.nodes['p'].data[f'f_junc_{suffix}'] = bg.nodes['p'].data['f_junc'].clone()

            
        update_funcs = {e:(fn.copy_e('h','m'), partial(self.msg_func, attn=self.attn[''.join(e)], field=f'f_{suffix}')) for e in self.homo_etypes }
        # update_funcs.update({e:(fn.copy_src(f'f_junc_{suffix}','mail'),partial(self.msg_func, attn=self.attn[''.join(e)], field=f'f_junc_{suffix}')) for e in self.hetero_etypes})
        # update_funcs.update({e:(fn.u_mul_e(f'f_junc_{suffix}',1.0,'m'),partial(self.msg_func, attn=self.attn[''.join(e)], field=f'f_junc_{suffix}')) for e in self.hetero_etypes})

        # message passing
        for i in range(self.depth-1):
            bg.multi_update_all(update_funcs,cross_reducer='sum')
            for e in self.hetero_etypes:
                apply_custom_copy_src(
                    bg,
                    e,
                    src_field=f'f_junc_{suffix}',
                    out_field='m',
                    reduce_func=partial(self.msg_func, attn=self.attn[''.join(e)], field=f'f_junc_{suffix}')
                )
            for edge_type in self.homo_etypes:
                bg.edges[edge_type].data['rev_h']=reverse_edge(bg.edges[edge_type].data['h'])
                bg.apply_edges(partial(del_reverse_message,field=f'f_{suffix}'),etype=edge_type)
                bg.apply_edges(partial(self.update_edge,layer=self.mp_list[''.join(edge_type)][i]), etype=edge_type)

        # last update of node feature
        update_funcs = {e:(fn.copy_e('h','mail'),partial(self.update_node,field=f'f_{suffix}',layer=self.node_last_layer[e[0]])) for e in self.homo_etypes}
        bg.multi_update_all(update_funcs,cross_reducer='sum')

        # last update of junc feature
        # bg.multi_update_all({e:(fn.copy_src(f'f_junc_{suffix}','mail'),
        #                          partial(self.update_node,field=f'f_junc_{suffix}',layer=self.node_last_layer['junc'])) for e in self.hetero_etypes},
        #                          cross_reducer='sum')
        # bg.multi_update_all({e:(fn.u_mul_e(f'f_junc_{suffix}',1.0,'mail'),
        #                     partial(self.update_node,field=f'f_junc_{suffix}',layer=self.node_last_layer['junc'])) for e in self.hetero_etypes},
        #                     cross_reducer='sum')
        # bg.multi_update_all({e:(partial(copy_src, src_field=f'f_junc_{suffix}', out_field='mail'),
        #                     partial(self.update_node,field=f'f_junc_{suffix}',layer=self.node_last_layer['junc'])) for e in self.hetero_etypes},
        #                     cross_reducer='sum')
        for e in self.hetero_etypes:
            apply_custom_copy_src(
                bg,
                e,
                src_field=f'f_junc_{suffix}',
                out_field='mail',
                reduce_func=partial(self.update_node, field=f'f_junc_{suffix}', layer=self.node_last_layer['junc'])
            )

class PharmHGT(nn.Module):
    def __init__(self, hid_dim, act, depth, atom_dim, bond_dim, pharm_dim, reac_dim):
        super(PharmHGT,self).__init__()
        # hid_dim = args['hid_dim']
        self.act = get_func(act)
        self.depth = depth
        self.output_dim = hid_dim
        # init
        # atom view
        self.w_atom = nn.Linear(atom_dim,hid_dim)
        self.w_bond = nn.Linear(bond_dim,hid_dim)
        # pharm view
        self.w_pharm = nn.Linear(pharm_dim,hid_dim)
        self.w_reac = nn.Linear(reac_dim,hid_dim)
        # junction view
        self.w_junc = nn.Linear(atom_dim + pharm_dim, hid_dim)

        ## define the view during massage passing
        # self.mp = MVMP(msg_func=add_attn,hid_dim=hid_dim,depth=self.depth,view='aj',suffix='h',act=self.act)
        # ('a','b','a'), ('p','r','p'), ('a','j','p'), ('p','j','a') are in apj view.
        # so there might no need to run 'a', 'ap' in advance.
        self.mp = MVMP(msg_func=add_attn,hid_dim=hid_dim,depth=self.depth,view='apj',suffix='h',act=self.act)
        # self.mp_junc = MVMP(msg_func=add_attn,hid_dim=hid_dim,depth=self.depth,view='apj',suffix='j',act=self.act)

        self.initialize_weights()

    def initialize_weights(self):
        for param in self.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                if self.act == nn.ReLU:
                    nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(param)

    def init_feature(self, bg):
        
        bg.nodes['a'].data['f'] = self.act(self.w_atom(bg.nodes['a'].data['f']))
        bg.edges[('a','b','a')].data['x'] = self.act(self.w_bond(bg.edges[('a','b','a')].data['x']))
        bg.nodes['p'].data['f'] = self.act(self.w_pharm(bg.nodes['p'].data['f']))
        if bg.edges[('p','r','p')].data['x'].numel():
            bg.edges[('p','r','p')].data['x'] = self.act(self.w_reac(bg.edges[('p','r','p')].data['x']))

        bg.nodes['a'].data['f_junc'] = self.act(self.w_junc(bg.nodes['a'].data['f_junc']))
        bg.nodes['p'].data['f_junc'] = self.act(self.w_junc(bg.nodes['p'].data['f_junc']))
        
    def forward(self,bg, finetune=False):
        """
        Args:
            bg: a batch of graphs
        """
        self.init_feature(bg)
        self.mp(bg)
        
        if finetune:
            return bg # return for GNN decoder.
        else:
            return self.encoder(bg) # if decoder is not MLP, then this is not needed, but a return for unified interface.

    def encoder(self, bg):
        """
        Baseline for pretrain.
        Extract embeddings directly from bg and do a simple cat.
        """
        embed_f_a = bg.nodes['a'].data['f_h']
        embed_f_p = bg.nodes['p'].data['f_h']
        # embed_junc_h_a = bg.nodes['a'].data['f_junc_h']
        # embed_junc_h_p = bg.nodes['p'].data['f_junc_h']

        # embed_a_fused = torch.mean(torch.stack([embed_f_a, embed_junc_h_a], dim=1), dim=1)
        # mbed_p_fused = torch.mean(torch.stack([embed_f_p, embed_junc_h_p], dim=1), dim=1)
        
        return embed_f_a, embed_f_p, bg
    
class PharmHGT_FP(nn.Module):
    """
    PharmHGT pretrained pooling. FP: From Pretrained.
    """
    def __init__(self, cfg, hid_dim, act, depth, atom_dim, bond_dim, pharm_dim, reac_dim, load_pretrained):
        super(PharmHGT_FP,self).__init__()
        self.cfg = cfg
        self.act = get_func(act)
        self.depth = depth
        self.output_dim = hid_dim
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.pharm_dim = pharm_dim
        self.reac_dim = reac_dim
        self.load_pretrained = load_pretrained

        if self.cfg.train.readout == 'att':
            self.readout = HierAttnReadout(hid_dim)
        elif self.cfg.train.readout == 'gru':
            self.readout = GRUReadout(hid_dim)
        elif self.cfg.train.readout == 'none':
            self.readout = None

        if self.cfg.train.pooling == 'mean':
            self.pooling = GlobalPooling('mean')
        elif self.cfg.train.pooling == 'sum':
            self.pooling = GlobalPooling('sum')
        elif self.cfg.train.pooling == 'max':
            self.pooling = GlobalPooling('max')
        else:
            raise ValueError(f"Invalid pooling method: {self.cfg.train.pooling}")
        
        # if readout, remove the pooling layer
        if self.readout:
            self.pooling = None
        
        self.mlp = MLP(hid_dim, num_classes=cfg.train.num_tasks)

        self.initialize_weights()
        
        self.pretrain_model = None

    def initialize_weights(self):
        for param in self.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                if self.act == nn.ReLU:
                    nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(param)

    def load_model(
            self,
            checkpoint_path: str,
            hid_dim: int,
            act: str,
            depth: int,
            atom_dim: int,
            bond_dim: int,
            pharm_dim: int,
            reac_dim: int,
            device: torch.device
        ):
        model = PharmHGT(
            hid_dim=hid_dim,
            act=act,
            depth=depth,
            atom_dim=atom_dim,
            bond_dim=bond_dim,
            pharm_dim=pharm_dim,
            reac_dim=reac_dim
        )
        model.to(device)

        if not self.load_pretrained:
            info("No pretrained model loaded. Start training from scratch.")
            return model
        
        # model save code
        # torch.save({
        #     'epoch': epoch,
        #     'encoder_state_dict': self.encoder.state_dict(),
        #     'decoder_state_dict': self.decoder.state_dict(),
        #     'optimizer_state_dict': self.optimizer.state_dict(),
        #     'metrics': metrics,
        # }, os.path.join(checkpoint_dir, "model.pt"))

        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict["encoder_state_dict"])

        # print model and estimate parameters
        info(f"loaded model from {checkpoint_path}, with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.") 

        return model
    
    def forward(self, bg):
        if not self.pretrain_model:
            # loading model
            model = self.load_model(
                checkpoint_path=self.cfg.train.checkpoint_path,
                hid_dim=self.output_dim,
                act=self.act,
                depth=self.depth,
                atom_dim=self.atom_dim,
                bond_dim=self.bond_dim,
                pharm_dim=self.pharm_dim,
                reac_dim=self.reac_dim,
                device=bg.device
            )
            self.pretrain_model = model

        # embedding extraction
        with torch.no_grad():
            bg = self.pretrain_model(bg, finetune=True)
            # embed_f_a = bg.nodes['a'].data['f_h'] torch.Size([26623, 300])
            # embed_f_p = bg.nodes['p'].data['f_h'] torch.Size([6775, 300])

        # if readout
        if self.readout:
            bg = self.readout(bg)

        # pooling
        if self.pooling:
            bg_embeds = self.pooling(bg)  # => shape [B, hid_dim]
        else:
            bg_embeds = bg # bg by att readout is a graph_embed

        # mlp
        logits = self.mlp(bg_embeds)  # => shape [B, num_classes]

        return logits # shape [B, 1] for binary classification
    
class MLP(nn.Module):
    """feed bg_embeds to MLP and give binary classification, note that num_classes=1"""
    def __init__(self, hid_dim, num_classes, readout):
        super(MLP,self).__init__()
        if readout == 'gru':  # bidirectional
            input_dim = 2 * hid_dim
        else:
            input_dim = hid_dim
            
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hid_dim),  # [600 → 300]
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim // 2),
            nn.ReLU(),
            nn.Linear(hid_dim//2, num_classes)
        )

    def forward(self, bg_embeds):
        return self.mlp(bg_embeds)
    
class GRUReadout(nn.Module):
    """
    Use explicit length-based Mask instead of judging whether the feature is 0.
    - Split the 'a' and 'p' nodes into (B, Lx, D) forms respectively;
    - Use node_size_a[i], node_size_p[i] to generate mask;
    - Perform cross-attention (a => p) or (p => a) in multi-head attention;
    - Then use GRU to process the sequence and aggregate to get graph-level embedding.
    """
    def __init__(self, hid_dim, bidirectional=True):
        super(Node_GRU, self).__init__()
        self.hid_dim = hid_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Cross-Attention: query=a, key=value=p
        self.cross_attn = MultiHeadedAttention(h=4, d_model=hid_dim)
        
        # GRU for sequences: [B, L, hid_dim * num_directions]
        self.gru = nn.GRU(hid_dim, hid_dim, batch_first=True, 
                          bidirectional=bidirectional)
        
    def _pad_nodes(self, bg, ntype, field, device):
        """
        Split the node features of ntype in bg into [B, maxL, hid_dim] and return:
        - padded_features: zero-padding for each graph
        - mask: indicates which positions are padding (True means padding, invalid)
        - node_sizes: vector of length B, the actual number of nodes in each graph
        """
        feat = bg.nodes[ntype].data[field]    # [total_num_nodes, hid_dim]
        node_sizes = bg.batch_num_nodes(ntype)  # [B] node number for each graph
        max_len = node_sizes.max().item()
        
        # prepare a tensor to store zero-padded features
        padded_features = feat.new_zeros((bg.batch_size, max_len, feat.size(-1)))
        
        # prepare a mask at the same time: [B, max_len], True indicates padding position
        # first default to all True, then mark the real node part as False
        mask = torch.ones((bg.batch_size, max_len), dtype=torch.bool, device=device)
        
        # cumulative index to align feats
        start_idx = 0
        for i in range(bg.batch_size):
            size_i = node_sizes[i].item()
            if size_i > 0:
                # take the node vector belonging to the i-th graph
                cur_feats = feat[start_idx : start_idx + size_i]      # [size_i, hid_dim]
                padded_features[i, :size_i] = cur_feats
                # set the mask position corresponding to the valid node part to False
                mask[i, :size_i] = False
            start_idx += size_i

        return padded_features, mask, node_sizes
    
    def forward(self, bg, suffix='h'):
        """
        1) Pad the 'a' and 'p' nodes to (B, La, D) / (B, Lp, D) respectively;
        2) Perform cross-attention: query=a, key/value=p;
        - Construct cross_mask to mask the padding part;
        3) Add the residual with the original feature (optional);
        4) Process the output with GRU;
        5) Perform mean or max aggregation on the sequence output according to node_sizes (such as p nodes) to obtain the graph-level vector.
        """
        device = bg.device
        
        # 1) get the features of nodes a and p and pad them
        a_padded, a_mask, a_sizes = self._pad_nodes(bg, 'a', f'f_{suffix}', device)
        p_padded, p_mask, p_sizes = self._pad_nodes(bg, 'p', f'f_{suffix}', device)
        # a_padded, p_padded: [B, Lmax, hid_dim]
        # a_mask, p_mask: [B, Lmax] (True => padding)
        
        # 2) construct cross_mask, shape [B, La, Lp], True positions are masked in attention
        # If position a is padding, or position p is padding, set to True
        # a_mask: (B, La), p_mask: (B, Lp)
        # cross_mask[b, i, j] = True if (a_mask[b,i] = True or p_mask[b,j] = True)
        B, La, D = a_padded.shape
        _, Lp, _ = p_padded.shape
        
        a_mask_3d = a_mask.unsqueeze(-1).expand(B, La, Lp)  # (B, La, Lp)
        p_mask_3d = p_mask.unsqueeze(1).expand(B, La, Lp)  # (B, La, Lp)
        cross_mask = a_mask_3d | p_mask_3d                 # element-wise OR
        
        # 3) do cross-attention: query=a, key=p, value=p
        # first align the mask dimensions to the attention requirements -> [B, La, Lp]
        a_updated = self.cross_attn(query=a_padded, key=p_padded, value=p_padded, mask=cross_mask)
        
        # residue: a_updated + a_padded
        a_updated = a_padded + a_updated
        
        # 4) process with GRU (B, La, D)
        # initialize hidden: [num_directions, B, hid_dim]
        hidden_0 = a_updated.mean(dim=1, keepdim=True)  # shape (B,1,D)
        hidden_0 = hidden_0.repeat(self.num_directions, 1, 1)  # [num_directions, B, D]
        
        # run GRU
        # a_out: [B, La, hid_dim * num_directions]
        a_out, hidden_n = self.gru(a_updated, hidden_0)
        
        # 5) aggregate a_out after unpadding (or aggregate directly on the padded tensor)
        # do mean pooling on a_out to get the embedding of each graph
        graph_embed = []
        for i in range(bg.batch_size):
            length_i = a_sizes[i].item()
            # If length_i=0, fill in 0
            if length_i == 0:
                graph_embed.append(a_out.new_zeros(1, self.num_directions * self.hid_dim))
            else:
                # a_out[i, :length_i], shape=(L_i, hid_dim * dir)
                g_i = a_out[i, :length_i].mean(dim=0)
                graph_embed.append(g_i.unsqueeze(0))

        graph_embed = torch.cat(graph_embed, dim=0)  # [B, hid_dim * num_directions]
        return graph_embed
    
class AmpHGT_FT(nn.Module):
    """
    AmpHGT (PharmHGT_pretrained + pooling/readout) trained. FT: From Trained.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass