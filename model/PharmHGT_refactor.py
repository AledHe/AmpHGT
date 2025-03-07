import torch
from torch import nn
from dgl import function as fn
from functools import partial

from model.utils import get_func
from utils.std_logger import error, info
from .readouts import GlobalPooling, HierAttnReadout, SemanticAttentionReadout, MultiHeadedAttention, GRUReadout
from .sequence_embedding import VOCAB_SIZE, ResidueEncoder
from .ESM2_ConvTrans_ensemble import ESM_CovT_Ensemble
from .ESM2 import ESMResidueEncoder

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

class NodeFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.gate_linear = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, feat_main, feat_junc):
        # feat_main, feat_junc: (N, hidden_dim)
        cat_feat = torch.cat([feat_main, feat_junc], dim=-1)  # (N, 2 * hidden_dim)
        gate = torch.sigmoid(self.gate_linear(cat_feat))      # (N, hidden_dim)
        fused = gate * feat_main + (1 - gate) * feat_junc
        fused = self.ln(fused)
        return fused

class MVMP(nn.Module):
    def __init__(self,msg_func=add_attn,hid_dim=512,depth=9,view='aba',suffix='h',act=nn.ReLU()):
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
        self.node_types = ['a']
        
        if 'p' in view:
            self.homo_etypes.append(('p','r','p'))
            self.node_types.append('p')
        
        if 'j' in view and 'a' in view and 'p' in view:
            self.node_types.append('junc') # junc is not a defined nodes, more like a bipartite edge.
            self.hetero_etypes.extend([('a','j','p'), ('p','j','a')])
        
        if 'r' in view:
            self.homo_etypes.append(('rsd','s','rsd'))
            self.node_types.append('rsd')
            if 'p' in view:
                self.hetero_etypes.extend([('p','l','rsd'), ('rsd','l','p')])

        info("nodes:", {n for n in self.node_types})
        info("home_etypes:", {e for e in self.homo_etypes})
        info("hetero_etypes:", {e for e in self.hetero_etypes})

        self.attn = nn.ModuleDict()
        for etype in self.homo_etypes + self.hetero_etypes:
            self.attn[''.join(etype)] = MultiHeadedAttention(4,hid_dim)

        self.mp_list = nn.ModuleDict()
        for edge_type in self.homo_etypes:
            self.mp_list[''.join(edge_type)] = nn.ModuleList([nn.Linear(hid_dim,hid_dim) for i in range(depth-1)])

        self.node_last_layer = nn.ModuleDict()
        for ntype in self.node_types:
            self.node_last_layer[ntype] = nn.Linear(3*hid_dim,hid_dim)

        # self.nodefusion = nn.ModuleDict({
        #     ntype: NodeFusion(hid_dim) 
        #     for ntype in ['a', 'p', 'rsd']
        # })

    def update_edge(self,edge,layer):
        return {'h':self.act(edge.data['x']+layer(edge.data['m']))}
    
    def _update_node(self,node,field,layer):
        return {field:layer(torch.cat([node.mailbox['mail'].sum(dim=1),
                                       node.data[field],
                                       node.data['f']],1))}

    def update_node(self, node, field, layer):
        """
        node : a DGL NodeBatch
        field: string of the node-data field we are updating (e.g. 'f_h')
        layer: a linear (or MLP) layer that maps from concat dimension -> hid_dim
        """
        # grab the *old* node feature that we want to skip from
        old_feat = node.data[field]

        # sum mailbox messages
        msg_sum = node.mailbox['mail'].sum(dim=1)

        concat_input = torch.cat([
            msg_sum,          # aggregated messages from neighbors
            old_feat,         # previous layer embedding
            node.data['f']    # raw or initial embedding
        ], dim=1)

        new_feat = layer(concat_input)

        # add the residual connection
        out_feat = new_feat + old_feat

        # apply nonlinearity
        out_feat = self.act(out_feat)  # or a LayerNorm etc.

        return { field: out_feat }
    
    def init_node(self,node):
        return {f'f_{self.suffix}':node.data['f'].clone()}

    def init_edge(self,edge):
        return {'h':edge.data['x'].clone()}

    def get_update_funcs(self, suffix):
        funcs = {}
        # homo
        for e in self.homo_etypes:
            funcs[e] = (
                fn.copy_e('h', 'm'),
                partial(self.msg_func, attn=self.attn[''.join(e)], field=f'f_{suffix}')
            )
        # hetero
        for e in self.hetero_etypes:
            src_type = e[0]
            funcs[e] = (
                fn.copy_u(f'f_junc_{suffix}', 'm'),
                partial(self.msg_func, attn=self.attn[''.join(e)], field=f'f_junc_{suffix}')
            )
        return funcs

    def forward(self,bg):
        suffix = self.suffix
        for ntype in self.node_types:
            if ntype != 'junc':
                bg.apply_nodes(self.init_node,ntype=ntype)
                
        for etype in self.homo_etypes:
            bg.apply_edges(self.init_edge,etype=etype)
        
        if 'j' in self.view:
            bg.nodes['a'].data[f'f_junc_{suffix}'] = bg.nodes['a'].data['f_junc'].clone()
            bg.nodes['p'].data[f'f_junc_{suffix}'] = bg.nodes['p'].data['f_junc'].clone()
            bg.nodes['rsd'].data[f'f_junc_{suffix}'] = bg.nodes['rsd'].data['f_junc'].clone()

        # update_funcs = {e:(fn.copy_e('h','m'), partial(self.msg_func, attn=self.attn[''.join(e)], field=f'f_{suffix}')) for e in self.homo_etypes }

        # message passing
        for i in range(self.depth-1):
            bg.multi_update_all(self.get_update_funcs(self.suffix),cross_reducer='sum')
            # for e in self.hetero_etypes:
            #     apply_custom_copy_src(
            #         bg,
            #         e,
            #         src_field=f'f_junc_{suffix}',
            #         out_field='m',
            #         reduce_func=partial(self.msg_func, attn=self.attn[''.join(e)], field=f'f_junc_{suffix}')
            #     )
            for edge_type in self.homo_etypes:
                bg.edges[edge_type].data['rev_h']=reverse_edge(bg.edges[edge_type].data['h'])
                bg.apply_edges(partial(del_reverse_message,field=f'f_{suffix}'),etype=edge_type)
                bg.apply_edges(partial(self.update_edge,layer=self.mp_list[''.join(edge_type)][i]), etype=edge_type)

        # last update of node feature
        # update_funcs = {e:(fn.copy_e('h','mail'),partial(self.update_node,field=f'f_{suffix}',layer=self.node_last_layer[e[0]])) for e in self.homo_etypes}
        # bg.multi_update_all(update_funcs,cross_reducer='sum')

        # for e in self.hetero_etypes:
        #     apply_custom_copy_src(
        #         bg,
        #         e,
        #         src_field=f'f_junc_{suffix}',
        #         out_field='mail',
        #         reduce_func=partial(self.update_node, field=f'f_junc_{suffix}', layer=self.node_last_layer['junc'])
        #     )
        final_update_funcs = {}
        for e in self.homo_etypes:
            dst_type = e[2]
            final_update_funcs[e] = (
                fn.copy_e('h', 'mail'),
                partial(self.update_node, field=f'f_{self.suffix}', layer=self.node_last_layer[dst_type])
            )
        for e in self.hetero_etypes:
            dst_type = e[2]
            final_update_funcs[e] = (
                fn.copy_u(f'f_junc_{self.suffix}', 'mail'),
                partial(self.update_node, field=f'f_junc_{self.suffix}', layer=self.node_last_layer['junc'])
            )
        bg.multi_update_all(final_update_funcs, cross_reducer='sum')
        
        # aftert the last message passing, fusion feature.
        # for ntype in ['a', 'p', 'rsd']:
        #     if bg.num_nodes(ntype) == 0:
        #         continue

        #     feat_main = bg.nodes[ntype].data.get(f'f_{self.suffix}')
        #     feat_junc = bg.nodes[ntype].data.get(f'f_junc_{self.suffix}')

        #     if feat_main is None or feat_junc is None:
        #         error("feat_main is None or feat_junc is None")
            
        #     fused_feat = self.nodefusion[ntype](feat_main, feat_junc)
        #     bg.nodes[ntype].data['f_final'] = fused_feat # (N, hid_dim)
        
class PharmHGT(nn.Module):
    def __init__(self, hid_dim, act, depth, atom_dim, bond_dim, pharm_dim, reac_dim, device, sq_embed):
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

        if sq_embed == "CovTrans":
            self.rsd_encoder = ResidueEncoder(
                vocab_size=VOCAB_SIZE,
                embed_dim=hid_dim,
                conv_channels=hid_dim,
                ff_dim=hid_dim * 4,
            ).to(device=device)
        elif sq_embed == "ESM2":
            self.rsd_encoder = ESMResidueEncoder().to(device)
            self.rsd_dim = ESMResidueEncoder().hidden_dim
        elif sq_embed == "Ensemble":
            self.rsd_encoder = ESM_CovT_Ensemble(hid_dim, device).to(device)

        # residue view
        if hasattr(self, 'rsd_dim'):
            self.w_rsd_proj = nn.Sequential(
                nn.Linear(self.rsd_dim, hid_dim * 2),
                nn.ReLU(),
                nn.Linear(hid_dim * 2, hid_dim),
                nn.ReLU(),
            )
        self.w_rsd_junc = nn.Linear(hid_dim, hid_dim)
        self.w_rsd_edge = nn.Linear(1, hid_dim)

        ## define the view during massage passing
        # self.mp = MVMP(msg_func=add_attn,hid_dim=hid_dim,depth=self.depth,view='aj',suffix='h',act=self.act)
        # ('a','b','a'), ('p','r','p'), ('a','j','p'), ('p','j','a') are in apj view.
        # so there might no need to run 'a', 'ap' in advance.
        self.mp = MVMP(msg_func=add_attn, hid_dim=hid_dim, depth=self.depth, view='apjr', suffix='h', act=self.act)
        # self.mp_junc = MVMP(msg_func=add_attn,hid_dim=hid_dim,depth=self.depth,view='apj',suffix='j',act=self.act)

        self.initialize_weights()

    def initialize_weights(self):
        for name, param in self.named_parameters():
            if 'rsd_encoder' in name:
                continue

            if param.dim() == 1:
                nn.init.constant_(param, 0.0)
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

        # init the rsd_junc here.
        # make sure the edge feature projects to the proper dimension.
        if bg.edges[('rsd','s','rsd')].data['x'].size(1) == 1:
            bg.edges[('rsd','s','rsd')].data['x'] = self.act(self.w_rsd_edge(bg.edges[('rsd','s','rsd')].data['x']))

        # init the rsd_junc here.
        if 'f_junc' not in bg.nodes['rsd'].data:
            dim_rsd = bg.nodes['rsd'].data['f'].size(1)
            # match the dim
            junction_dim = bg.nodes['a'].data['f_junc'].size(1)
            padding_dim = junction_dim - dim_rsd
            if padding_dim > 0:
                bg.nodes['rsd'].data['f_junc'] = torch.cat([
                    torch.zeros(bg.num_nodes('rsd'), padding_dim, 
                               device=bg.nodes['rsd'].data['f'].device),
                    bg.nodes['rsd'].data['f']
                ], 1)
            else:
                bg.nodes['rsd'].data['f_junc'] = bg.nodes['rsd'].data['f'].clone()

        bg.nodes['rsd'].data['f_junc'] = self.act(self.w_rsd_junc(bg.nodes['rsd'].data['f_junc']))
        
    def forward(self,bg, finetune=False):
        """
        Args:
            bg: a batch of graphs
        """
        rsd_emb, cls_emb = self.rsd_encoder(bg) # (num_nodes, conv_channels), (batch_size, conv_channels)

        # for esm, we need to project hidden_dim.
        if self.w_rsd_proj:
            rsd_emb = self.w_rsd_proj(rsd_emb)  # => [total_num_rsd, hid_dim]

        bg.nodes['rsd'].data['f'] = rsd_emb # apply new added residue nodes feature by sequence encoder.

        self.init_feature(bg)
        self.mp(bg)
        
        if finetune:
            return bg, cls_emb
        else:
            return self.encoder(bg, cls_emb)

    def encoder(self, bg, cls_emb):
        """
        Baseline for pretrain.
        Extract embeddings directly from bg and do a simple cat.
        """
        embed_f_a = bg.nodes['a'].data['f_h']
        embed_f_p = bg.nodes['p'].data['f_h']
        embed_f_rsd = bg.nodes['rsd'].data['f_h']
        # embed_f_a, embed_f_p, embed_f_rsd: (N, hid_dim), N is the number of nodes in the graph.
        # torch.Size([1530, 512]) torch.Size([384, 512]) torch.Size([179, 512])
        
        return embed_f_a, embed_f_p, embed_f_rsd, bg, cls_emb
    
class PharmHGT_FP(nn.Module):
    """
    PharmHGT pretrained pooling. FP: From Pretrained.
    """
    def __init__(self, cfg, hid_dim, act, depth, atom_dim, bond_dim, pharm_dim, reac_dim, device, sq_embed, load_pretrained):
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

        if self.cfg.train.readout == 'har':
            self.readout = HierAttnReadout(hid_dim)
        elif self.cfg.train.readout == 'gru':
            self.readout = GRUReadout(hid_dim)
        elif self.cfg.train.readout == 'sar':
            self.readout = SemanticAttentionReadout(hid_dim)
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

        if self.cfg.train.fusion == 'gate':
            self.fusion = GatedFusion(embed_dim=hid_dim)
        elif self.cfg.train.fusion == 'bilinear':
            self.fusion = BilinearFusion(embed_dim=hid_dim)
        elif self.cfg.train.fusion == 'attention':
            self.fusion = BilinearAttentionFusion(d_model=hid_dim, num_heads=4)
        else:
            raise ValueError(f"Invalid pooling method: {self.cfg.train.fusion}")
            
        self.mlp = MLP(hid_dim=hid_dim, num_classes=cfg.train.num_tasks, input_dim=hid_dim)

        self.pretrain_model = self.load_model(
            checkpoint_path=self.cfg.train.checkpoint_path,
            hid_dim=self.output_dim,
            act=self.act,
            depth=self.depth,
            atom_dim=self.atom_dim,
            bond_dim=self.bond_dim,
            pharm_dim=self.pharm_dim,
            reac_dim=self.reac_dim,
            device=device,
            sq_embed=sq_embed
        )
        
        if sq_embed == "ESM2":
            self.esm_proj = nn.Sequential(
                nn.Linear(1280, hid_dim * 2), # hard coded here...
                nn.ReLU(),
                nn.Linear(hid_dim * 2, hid_dim),
                nn.ReLU()
            )

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
            device: torch.device,
            sq_embed: str
        ):
        model = PharmHGT(
            hid_dim=hid_dim,
            act=act,
            depth=depth,
            atom_dim=atom_dim,
            bond_dim=bond_dim,
            pharm_dim=pharm_dim,
            reac_dim=reac_dim,
            device=device,
            sq_embed = sq_embed
        ).to(device)

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
        # embedding extraction
        if self.load_pretrained and not self.cfg.train.full_ft:
            # do not mess up the pretrained model
            with torch.no_grad():
                bg, cls_emb = self.pretrain_model(bg, finetune=True) # cls_emb: (batch_size, conv_channels), conv_channels = hid_dim
        else:
            bg, cls_emb = self.pretrain_model(bg, finetune=True)

        # if readout, or pooling
        if self.readout:
            bg_embeds = self.readout(bg)   # if GRUReadout, (batch_size, hid_dim*2)
        else:
            bg_embeds = self.pooling(bg)   # (batch_size, hid_dim)

        # print(bg_embeds.size())
        # print(cls_emb.size())
        # torch.Size([128, 300])
        # torch.Size([128, 1280])
        if self.esm_proj:
            cls_emb = self.esm_proj(cls_emb)

        fused_feats = self.fusion(bg_embeds, cls_emb)  # (batch_size, hid_dim)

        # cat_feats = torch.cat([bg_embeds, cls_emb], dim=-1)

        # info("bg_embeds均值:", bg_embeds.mean().item())
        # info("cls_emb均值:", cls_emb.mean().item())
        # info("融合后特征均值:", fused_feats.mean().item(), st=5)

        # mlp
        logits = self.mlp(fused_feats)  # => shape [B, num_classes]

        return logits # shape [B, 1] for binary classification
    
# bilinear is the better option.

class GatedFusion(nn.Module):
    def __init__(self, embed_dim):
        super(GatedFusion, self).__init__()
        self.gate_fc = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, feat1, feat2):
        """
        feat1, feat2: (batch_size, D)
        """
        # concat and gate
        cat_feat = torch.cat([feat1, feat2], dim=-1)  # (B, 2D)
        gate = torch.sigmoid(self.gate_fc(cat_feat))  # (B, D)
        out = gate * feat1 + (1.0 - gate) * feat2
        return out
    
class BilinearFusion(nn.Module):
    def __init__(self, embed_dim):
        super(BilinearFusion, self).__init__()
        self.linear = nn.Linear(embed_dim * embed_dim, embed_dim)
    
    def forward(self, feat1, feat2):
        # feat1, feat2: (B, D)
        outer_product = torch.bmm(feat1.unsqueeze(2), feat2.unsqueeze(1))
        B, D, _ = outer_product.size()
        fused = outer_product.view(B, -1)
        fused = self.linear(fused)
        return fused
    
class BilinearAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(BilinearAttentionFusion, self).__init__()
        self.bilinear = BilinearFusion(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
    def forward(self, feat_seq, feat_struct):
        bilinear_out = self.bilinear(feat_seq, feat_struct)
        bilinear_out = self.norm(bilinear_out)
        
        query = bilinear_out.unsqueeze(1)  # (B, 1, D)
        kv = torch.cat([feat_seq.unsqueeze(1), feat_struct.unsqueeze(1)], dim=1)  # (B, 2, D)
        attn_out, weights = self.attention(query, kv, kv)
        
        return attn_out.squeeze(1), weights
    
class MLP(nn.Module):
    """feed bg_embeds to MLP and give binary classification, note that num_classes=1"""
    def __init__(self, hid_dim, num_classes, input_dim):
        super(MLP,self).__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(input_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim // 2),
            nn.ReLU(),
            nn.Linear(hid_dim//2, num_classes)
        )

    def forward(self, bg_embeds):
        return self.mlp(bg_embeds)
    
class AmpHGT_FT(PharmHGT_FP):
    """
    AmpHGT (PharmHGT_pretrained + pooling/readout) trained. FT: From Trained.
    Inference-optimized version with inheritance

    1. Force load_pretrained=False (since weights are in state_dict)
    2. Add inference-specific forward
    3. Provide loading interface
    """
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args, 
            load_pretrained=False,
            **kwargs
        )

    def forward(self, bg):
        """ Inference-optimized forward """
        with torch.no_grad():
            # call the parent forward
            logits = super().forward(bg)
            return logits
        
    @classmethod
    def from_checkpoint(cls, checkpoint_path, cfg, device, **kwargs):
        model = cls(
            cfg=cfg,
            hid_dim=cfg.train.hid_dim,
            act=cfg.train.act,
            depth=cfg.train.depth,
            atom_dim=cfg.train.atom_dim,
            bond_dim=cfg.train.bond_dim,
            pharm_dim=cfg.train.pharm_dim,
            reac_dim=cfg.train.reac_dim,
            device=device,
            **kwargs
        )

        checkpoint = torch.load(checkpoint_path, map_location=device)

        state_dict = {
            k.replace('module.', ''): v  # if DDP, remove the 'module.' prefix
            for k, v in checkpoint['model_state_dict'].items()
        }
        model.load_state_dict(state_dict, strict=True)
        info(f"loaded model from {checkpoint_path}, with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.") 
        model.eval()
        return model