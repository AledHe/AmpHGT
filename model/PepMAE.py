"""
Customized prediction head, changed name to PepMAE.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from utils.features import ATOM_FEATURES

class PepMAE(nn.Module):
    """
    Peptide Masked Autoencoder for heterogeneous molecular graphs.
    Designed to work with PharmHGT and existing peptide data structure.
    """
    def __init__(self, 
                 hidden_dim: int,
                 atom_feat_dim: int,
                 fragment_feat_dim: int,
                 num_atom_classes: int,   # For atom classification
                 num_fragment_classes: int, # From vocab size
                 ):
        super().__init__()
        
        # Atom reconstruction components
        self.atom_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.atom_heads = nn.ModuleDict({
            'atom_features': nn.Linear(hidden_dim // 2, atom_feat_dim),
            'junction_features': nn.Linear(hidden_dim // 2, atom_feat_dim + fragment_feat_dim)
        })
        
        # Fragment reconstruction components
        self.fragment_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.fragment_heads = nn.ModuleDict({
            'fragment_features': nn.Linear(hidden_dim // 2, fragment_feat_dim),
            'junction_features': nn.Linear(hidden_dim // 2, atom_feat_dim + fragment_feat_dim),
            'maccs': nn.Linear(hidden_dim // 2, 167),  # MACCS keys dimension
            'pharm': nn.Linear(hidden_dim // 2, 27)    # Pharmacophore features dimension
        })

        # Classification components
        self.atom_classifier = nn.ModuleDict({
            'atomic_num': nn.Linear(hidden_dim, num_atom_classes),
        })
        
        self.fragment_classifier = nn.Linear(hidden_dim, num_fragment_classes)

    def forward(self, 
                atom_repr: torch.Tensor,
                fragment_repr: torch.Tensor,
                atom_mask: torch.Tensor,
                fragment_mask: torch.Tensor
                ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass compatible with existing PharmHGT output format
        """
        # Get masked node representations
        masked_atom_repr = atom_repr[atom_mask]
        masked_frag_repr = fragment_repr[fragment_mask]
        
        # 1. Feature Reconstruction
        # Decode atom features
        atom_decoded = self.atom_decoder(masked_atom_repr)
        
        # Decode fragment features
        frag_decoded = self.fragment_decoder(masked_frag_repr)

        reconstruction_predictions = {
            'atom': {
                name: head(atom_decoded)
                for name, head in self.atom_heads.items()
            },
            'fragment': {
                name: head(frag_decoded)
                for name, head in self.fragment_heads.items()
            }
        }
        
        # 2. Classification
        classification_predictions = {
            'atom': {
                'atomic_num': self.atom_classifier['atomic_num'](masked_atom_repr)
            },
            'fragment': {
                'vocab_idx': self.fragment_classifier(masked_frag_repr)
            }
        }
        
        return reconstruction_predictions, classification_predictions

class PepMAELoss(nn.Module):
    """
    Loss function aligned with existing metrics and feature structure
    """
    def __init__(self, 
                 reconstruction_weight: float = 1.0,
                 classification_weight: float = 1.0):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.classification_weight = classification_weight

        # reconstruction losses
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')

        # classification losses
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self,
                reconstruction_pred: Dict,
                reconstruction_true: Dict,
                classification_pred: Dict,
                classification_true: Dict
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        losses = {}
        metrics = {}
        
        # 1. Reconstruction losses
        recon_loss = 0.0
        for feat_type in ['atom', 'fragment']:
            for feat_name, pred in reconstruction_pred[feat_type].items():
                true = reconstruction_true[feat_type][feat_name]
                if feat_name in ['maccs', 'pharm']:
                    loss = self.bce(pred, true).mean()
                else:
                    loss = self.mse(pred, true).mean()
                losses[f'{feat_type}_{feat_name}_recon_loss'] = loss
                recon_loss += loss
        
        # 2. Classification losses
        class_loss = 0.0
        # Atom number classification
        atom_num_pred = classification_pred['atom']['atomic_num']
        atom_num_true = classification_true['atom']['atomic_num']
        atom_num_loss = self.ce(atom_num_pred, atom_num_true)
        losses['atom_num_class_loss'] = atom_num_loss
        
        # Fragment vocabulary classification
        frag_vocab_pred = classification_pred['fragment']['vocab_idx']
        frag_vocab_true = classification_true['fragment']['vocab_idx']
        frag_vocab_loss = self.ce(frag_vocab_pred, frag_vocab_true)
        losses['fragment_vocab_class_loss'] = frag_vocab_loss
        
        class_loss = atom_num_loss + frag_vocab_loss
        
        # Calculate accuracies
        atom_num_acc = (atom_num_pred.argmax(dim=1) == atom_num_true).float().mean()
        frag_vocab_acc = (frag_vocab_pred.argmax(dim=1) == frag_vocab_true).float().mean()
        
        metrics['atom_num_accuracy'] = atom_num_acc.item()
        metrics['fragment_vocab_accuracy'] = frag_vocab_acc.item()
        
        # Combine losses with weights
        total_loss = (self.reconstruction_weight * recon_loss + 
                     self.classification_weight * class_loss)
        
        losses['reconstruction_loss'] = recon_loss.item()
        losses['classification_loss'] = class_loss.item()
        losses['total_loss'] = total_loss.item()
        
        # Combine all metrics
        metrics.update(losses)
        
        return total_loss, metrics