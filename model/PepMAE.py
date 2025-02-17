"""
Customized prediction head, changed name to PepMAE.
"""

import dgl
import dgl.nn.pytorch as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class MLPDecoder(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 atom_feat_dim: int,
                 fragment_feat_dim: int,
                 num_atom_classes: int,
                 num_fragment_classes: int
                 ):
        super().__init__()
        self.atom_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, atom_feat_dim),
            nn.ReLU()
        )
        self.fragment_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fragment_feat_dim),
            nn.ReLU()
        )
        self.atom_classifier = nn.Linear(hidden_dim, num_atom_classes)
        self.fragment_classifier = nn.Linear(hidden_dim, num_fragment_classes)

    def forward(self,
                bg, 
                atom_mask: torch.Tensor,
                fragment_mask: torch.Tensor
                ):
        atom_decoded = self.atom_decoder(atom_mask)
        fragment_decoded = self.fragment_decoder(fragment_mask)

        atom_class = self.atom_classifier(atom_mask)
        fragment_class = self.fragment_classifier(fragment_mask)

        feature_decodes = {"atom": atom_decoded, "fragment": fragment_decoded}
        class_decodes = {"atom": atom_class, "fragment": fragment_class}

        return feature_decodes, class_decodes

class HeteroConv(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 atom_feat_dim: int,
                 fragment_feat_dim: int,
                 num_atom_classes: int,
                 num_fragment_classes: int,
                 num_layers = 2
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
            }, aggregate='mean')  # or 'sum', 'max', etc.
            self.convs.append(conv)

            # Layer Normalization
            self.norms.append(nn.ModuleDict({
                'a': nn.LayerNorm(hidden_dim),
                'p': nn.LayerNorm(hidden_dim)
            }))

        # 2) decode layer
        self.atom_feat_decoder = nn.Sequential(
            nn.Linear(hidden_dim, atom_feat_dim)
        )
        self.frag_feat_decoder = nn.Sequential(
            nn.Linear(hidden_dim, fragment_feat_dim)
        )

        # 3) classification layer
        self.atom_classifier = nn.Linear(hidden_dim, num_atom_classes)
        self.frag_classifier = nn.Linear(hidden_dim, num_fragment_classes)

    def forward(self, 
                bg: dgl.DGLHeteroGraph,
                ):
        """
        param:
        bg: current batch of graphs
        """
        # 1) Heterogeneous Convolution
        h = {
            'a': bg.nodes['a'].data['f_h'],
            'p': bg.nodes['p'].data['f_h']
        }

        for i in range(self.num_layers):
            # message passing
            h_update = self.convs[i](bg, h)
            
            # residual connection + layer normalization
            h = {
                'a': self.norms[i]['a'](h['a'] + F.relu(h_update['a'])),
                'p': self.norms[i]['p'](h['p'] + F.relu(h_update['p']))
            }

        feature_decodes = {
            'atom': self.atom_feat_decoder(h['a'][bg.nodes['a'].data['mask'] == True]),
            'fragment': self.frag_feat_decoder(h['p'][bg.nodes['p'].data['mask'] == True])
        }
        
        class_decodes = {
            'atom': self.atom_classifier(h['a'][bg.nodes['a'].data['mask'] == True]),
            'fragment': self.frag_classifier(h['p'][bg.nodes['p'].data['mask'] == True])
        }

        return feature_decodes, class_decodes

class HeteroAttNet(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 atom_feat_dim: int,
                 fragment_feat_dim: int,
                 num_atom_classes: int,
                 num_fragment_classes: int,
                 num_layers=2,
                 num_heads=4
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.num_layers = num_layers

        for _ in range(num_layers):
            conv = dglnn.HeteroGraphConv({
                ('a', 'b', 'a'): dglnn.GATConv(hidden_dim, self.head_dim, num_heads),
                ('p', 'r', 'p'): dglnn.GATConv(hidden_dim, self.head_dim, num_heads),
                ('a', 'j', 'p'): dglnn.GATConv(hidden_dim, self.head_dim, num_heads),
                ('p', 'j', 'a'): dglnn.GATConv(hidden_dim, self.head_dim, num_heads)
            }, aggregate='mean')
            self.convs.append(conv)

            # Layer Normalization
            self.norms.append(nn.ModuleDict({
                'a': nn.LayerNorm(hidden_dim),
                'p': nn.LayerNorm(hidden_dim)
            }))

        # Decoder layers
        self.atom_feat_decoder = nn.Linear(hidden_dim, atom_feat_dim)
        self.frag_feat_decoder = nn.Linear(hidden_dim, fragment_feat_dim)

        # Classifiers
        self.atom_classifier = nn.Linear(hidden_dim, num_atom_classes)
        self.frag_classifier = nn.Linear(hidden_dim, num_fragment_classes)

    def forward(self, bg):
        h = {
            'a': bg.nodes['a'].data['f_h'],
            'p': bg.nodes['p'].data['f_h']
        }

        for i in range(self.num_layers):
            # Apply GAT convolution
            h_update = self.convs[i](bg, h)
            
            # Residual connection and layer norm
            h = {
                'a': self.norms[i]['a'](h['a'] + F.relu(h_update['a'])),
                'p': self.norms[i]['p'](h['p'] + F.relu(h_update['p']))
            }

        # Decode features and classify
        feature_decodes = {
            'atom': self.atom_feat_decoder(h['a'][bg.nodes['a'].data['mask'] == True]),
            'fragment': self.frag_feat_decoder(h['p'][bg.nodes['p'].data['mask'] == True])
        }

        class_decodes = {
            'atom': self.atom_classifier(h['a'][bg.nodes['a'].data['mask'] == True]),
            'fragment': self.frag_classifier(h['p'][bg.nodes['p'].data['mask'] == True])
        }

        return feature_decodes, class_decodes

class PepMAE(nn.Module):
    """
    Controller for different decoders.
    """
    def __init__(self, 
                 hidden_dim: int,
                 atom_feat_dim: int,
                 fragment_feat_dim: int,
                 num_atom_classes: int,   # For atom classification
                 num_fragment_classes: int, # From vocab size
                 decoder_type = "MLP"
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.atom_feat_dim = atom_feat_dim
        self.fragment_feat_dim = fragment_feat_dim
        self.num_atom_classes = num_atom_classes
        self.num_fragment_classes = num_fragment_classes
        self.decoder_type = decoder_type

        if decoder_type == "MLP":
            self.decoder = MLPDecoder(hidden_dim, atom_feat_dim, fragment_feat_dim, num_atom_classes, num_fragment_classes)
        elif decoder_type == "HGC":
            self.decoder = HeteroConv(hidden_dim, atom_feat_dim, fragment_feat_dim, num_atom_classes, num_fragment_classes)
        elif decoder_type == "HAN":
            self.decoder = HeteroAttNet(hidden_dim, atom_feat_dim, fragment_feat_dim, num_atom_classes, num_fragment_classes)
        else:
            raise ValueError(f"Decoder type {decoder_type} not supported.")

    def forward(self,
                bg: dgl.DGLHeteroGraph,
                atom_mask: torch.Tensor,
                fragment_mask: torch.Tensor
                ):
        if self.decoder_type == "MLP":
            return self.decoder(bg, atom_mask, fragment_mask)
        else:
            return self.decoder(bg)

class PepMAELoss(nn.Module):
    """
    Loss function aligned with existing metrics and feature structure
    """
    def __init__(self, 
                 reconstruction_weight: float = 1.0,
                 classification_weight: float = 1.0,
                 alpha: float = 1.0, 
                 ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.classification_weight = classification_weight
        self.alpha = alpha

        self.mse = nn.MSELoss()

        # classification losses
        self.ce = nn.CrossEntropyLoss()

    def sce_loss(self, x, y):
        """Scaled Cosine Error (SCE) loss implementation from GraphMAE"""
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)
        loss = (1 - (x * y).sum(dim=-1)).pow_(self.alpha)
        return loss.mean()
    
    def compute_accuracy(self, pred, target):
        return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)
    
    def loss(self,
                feature_decodes: Dict[str, torch.Tensor],
                feature_targets: Dict[str, torch.Tensor],
                class_decodes: Dict[str, torch.Tensor],
                class_targets: Dict[str, torch.Tensor]
                ):
        # Reconstruction loss
        # print(feature_decodes["atom"].shape, feature_targets["atom"].shape)
        # print(feature_decodes["fragment"].shape, feature_targets["fragment"].shape)
        atom_recon_loss = self.sce_loss(feature_decodes["atom"], feature_targets["atom"])
        frag_recon_loss = self.sce_loss(feature_decodes["fragment"], feature_targets["fragment"])
        recon_loss = atom_recon_loss + frag_recon_loss

        # Classification loss
        # print(class_decodes["atom"], class_targets["atom"])
        # print(class_decodes["atom"].shape, class_targets["atom"].shape)
        # print(class_decodes["fragment"].shape, class_targets["fragment"].shape)
        atom_class_loss = self.ce(class_decodes["atom"].double(), class_targets["atom"][:, 0])
        frag_class_loss = self.ce(class_decodes["fragment"].double(), class_targets["fragment"])
        class_loss = atom_class_loss + frag_class_loss

        total_loss = self.reconstruction_weight * recon_loss \
                   + self.classification_weight * class_loss
        
        atom_acc = self.compute_accuracy(class_decodes["atom"], class_targets["atom"][:, 0])
        frag_acc = self.compute_accuracy(class_decodes["fragment"], class_targets["fragment"])

        batch_metrics = {
            "loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "class_loss": class_loss.item(),
            "atom_acc": atom_acc,
            "fragment_acc": frag_acc
        }

        return total_loss, batch_metrics
    
    def forward(self,
                feature_decodes: Dict[str, torch.Tensor],
                feature_targets: Dict[str, torch.Tensor],
                class_decodes: Dict[str, torch.Tensor],
                class_targets: Dict[str, torch.Tensor]
                ):
        return self.loss(feature_decodes, feature_targets, class_decodes, class_targets)