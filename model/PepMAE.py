"""
Customized prediction head, changed name to PepMAE.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

class PepMAE(nn.Module):
    """
    Peptide Masked Autoencoder for heterogeneous molecular graphs.
    Designed to work with PharmHGT and existing peptide data structure.
    """
    def __init__(self, 
                 hidden_dim: int,
                 atom_feat_dim: int,
                 fragment_feat_dim: int):
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

    def forward(self, 
                atom_repr: torch.Tensor,
                fragment_repr: torch.Tensor,
                atom_mask: torch.Tensor,
                fragment_mask: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass compatible with existing PharmHGT output format
        """
        # Get masked node representations
        masked_atom_repr = atom_repr[atom_mask]
        masked_frag_repr = fragment_repr[fragment_mask]
        
        # Decode atom features
        atom_decoded = self.atom_decoder(masked_atom_repr)
        atom_predictions = {
            name: head(atom_decoded)
            for name, head in self.atom_heads.items()
        }
        
        # Decode fragment features
        frag_decoded = self.fragment_decoder(masked_frag_repr)
        fragment_predictions = {
            name: head(frag_decoded)
            for name, head in self.fragment_heads.items()
        }
        
        return atom_predictions, fragment_predictions

class PepMAELoss(nn.Module):
    """
    Loss function aligned with existing metrics and feature structure
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.mse = nn.MSELoss(reduction='none')
        
    def forward(self,
                atom_pred: Dict[str, torch.Tensor],
                atom_true: Dict[str, torch.Tensor],
                frag_pred: Dict[str, torch.Tensor],
                frag_true: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        losses = {}
        
        # Atom reconstruction losses
        for feat_name, pred in atom_pred.items():
            true = atom_true[feat_name]
            if feat_name == 'junction_features':
                loss = self.mse(pred, true).mean()
            else:
                loss = self.bce(pred, true).mean()
            losses[f'atom_{feat_name}_loss'] = loss
            
        # Fragment reconstruction losses
        for feat_name, pred in frag_pred.items():
            true = frag_true[feat_name]
            if feat_name in ['maccs', 'pharm']:
                loss = self.bce(pred, true).mean()
            else:
                loss = self.mse(pred, true).mean()
            losses[f'fragment_{feat_name}_loss'] = loss
            
        # Compute total loss
        total_loss = sum(loss for loss in losses.values())
        losses['total_loss'] = total_loss
        
        return total_loss, losses