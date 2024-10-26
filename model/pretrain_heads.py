"""
Customized prediction head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from utils.features import ATOM_FEATURES

class AtomReconstructionHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        # Calculate total atom feature dimension
        self.feature_dims = {
            'atomic_num': len(ATOM_FEATURES['atomic_num']) + 1,  # +1 for unknown
            'degree': len(ATOM_FEATURES['degree']) + 1,
            'formal_charge': len(ATOM_FEATURES['formal_charge']) + 1,
            'chiral_tag': len(ATOM_FEATURES['chiral_tag']) + 1,
            'num_Hs': len(ATOM_FEATURES['num_Hs']) + 1,
            'hybridization': len(ATOM_FEATURES['hybridization']) + 1,
            'is_aromatic': 1,
            'mass': 1
        }
        self.total_feature_dim = sum(self.feature_dims.values())
        
        self.reconstruction_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.total_feature_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x shape: (num_masked_atoms, hidden_dim)
        features = self.reconstruction_net(x)
        
        # Split features into different properties and apply appropriate activations
        start = 0
        reconstructed_features = {}
        
        for name, dim in self.feature_dims.items():
            if name in ['is_aromatic', 'mass']:
                # Single value features use sigmoid
                reconstructed_features[name] = torch.sigmoid(features[:, start:start+dim])
            else:
                # One-hot encoded features use softmax
                reconstructed_features[name] = F.softmax(features[:, start:start+dim], dim=1)
            start += dim
            
        return reconstructed_features

class FragmentReconstructionHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        # Fragment feature dimensions from GetMaskFragmentFeats
        self.maccs_dim = 167  # MACCS keys dimension
        self.pharm_dim = 27   # Pharmacophore features dimension
        self.additional_bits = 2  # Two additional indicator bits
        
        self.reconstruction_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.maccs_dim + self.pharm_dim + self.additional_bits)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x shape: (num_masked_fragments, hidden_dim)
        features = self.reconstruction_net(x)
        
        return {
            'maccs': torch.sigmoid(features[:, :self.maccs_dim]),
            'pharm': torch.sigmoid(features[:, self.maccs_dim:self.maccs_dim + self.pharm_dim]),
            'indicators': torch.sigmoid(features[:, -self.additional_bits:])
        }

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss(reduction='none')
        
    def forward(self, 
                atom_pred: Dict[str, torch.Tensor], 
                atom_target: torch.Tensor,
                fragment_pred: Dict[str, torch.Tensor], 
                fragment_target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        losses = {}
        total_loss = 0.0
        
        # Atom reconstruction loss
        start = 0
        for name, pred_tensor in atom_pred.items():
            dim = pred_tensor.size(1)
            target = atom_target[:, start:start+dim]
            
            loss = self.bce_loss(pred_tensor, target).mean()
            losses[f'atom_{name}_loss'] = loss.item()
            total_loss += loss
            
            # Calculate accuracy for one-hot encoded features
            if name not in ['is_aromatic', 'mass']:
                acc = ((pred_tensor.argmax(dim=1) == target.argmax(dim=1)).float().mean())
                losses[f'atom_{name}_acc'] = acc.item()
            
            start += dim
            
        # Fragment reconstruction loss
        start = 0
        for name, pred_tensor in fragment_pred.items():
            dim = pred_tensor.size(1)
            target = fragment_target[:, start:start+dim]
            
            loss = self.bce_loss(pred_tensor, target).mean()
            losses[f'fragment_{name}_loss'] = loss.item()
            total_loss += loss
            
            # Calculate similarity for binary features using cosine similarity
            similarity = F.cosine_similarity(pred_tensor, target).mean()
            losses[f'fragment_{name}_similarity'] = similarity.item()
            
            start += dim
            
        losses['total_loss'] = total_loss.item()
        
        return total_loss, losses

class PretrainingHeads(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.atom_head = AtomReconstructionHead(hidden_dim)
        self.fragment_head = FragmentReconstructionHead(hidden_dim)
        
    def forward(self, atom_reps: torch.Tensor, fragment_reps: torch.Tensor, 
                atom_mask: torch.Tensor, fragment_mask: torch.Tensor):
        """
        Args:
            atom_reps: Atom representations from PharmHGT (N_atoms, hidden_dim)
            fragment_reps: Fragment representations from PharmHGT (N_fragments, hidden_dim)
            atom_mask: Boolean mask for atoms (N_atoms)
            fragment_mask: Boolean mask for fragments (N_fragments)
        Returns:
            Dict containing reconstructed features for atoms and fragments
        """
        atom_predictions = self.atom_head(atom_reps[atom_mask])
        fragment_predictions = self.fragment_head(fragment_reps[fragment_mask])
        
        return atom_predictions, fragment_predictions
    
class PretrainingMetrics:
    def __init__(self):
        self.feature_dims = {
            'atomic_num': len(ATOM_FEATURES['atomic_num']) + 1,
            'degree': len(ATOM_FEATURES['degree']) + 1,
            'hybridization': len(ATOM_FEATURES['hybridization']) + 1,
            'formal_charge': len(ATOM_FEATURES['formal_charge']) + 1
        }
    
    def compute_metrics(self, pred_atoms, true_atoms, pred_frags, true_frags):
        metrics = {}
        
        # 1. Chemical Property Recovery
        property_metrics = self._compute_chemical_property_metrics(
            pred_atoms, true_atoms, self.feature_dims
        )
        metrics.update(property_metrics)
        
        # 2. Fragment Pattern Recognition
        fragment_metrics = self._compute_fragment_metrics(
            pred_frags, true_frags
        )
        metrics.update(fragment_metrics)
        
        # 3. Representation Quality Metrics
        rep_metrics = self._compute_representation_metrics(
            pred_atoms, pred_frags
        )
        metrics.update(rep_metrics)
        
        # Calculate total loss
        total_loss = sum(value for key, value in metrics.items() 
                        if 'loss' in key or 'accuracy' in key)
        
        return total_loss, metrics

    def _compute_chemical_property_metrics(self, pred, target, feature_dims):
        """Focus on chemically significant properties"""
        metrics = {}
        start = 0
        for name, dim in feature_dims.items():
            # Using pred as dictionary correctly
            pred_feat = pred[name]  # This works
            target_feat = target[:, start:start+dim]
            
            accuracy = (pred_feat.argmax(1) == target_feat.argmax(1)).float().mean()
            metrics[f'{name}_accuracy'] = accuracy
            
            similarity = F.cosine_similarity(
                F.softmax(pred_feat, dim=1),
                target_feat,
                dim=1
            ).mean()
            metrics[f'{name}_chemical_sim'] = similarity
            
            start += dim
        return metrics

    def _compute_fragment_metrics(self, pred_frags, true_frags):
        """Measure fragment pattern recognition"""
        # MACCS keys comparison (important for structural similarity)
        maccs_similarity = F.cosine_similarity(
            pred_frags['maccs'],
            true_frags[:, :167],
            dim=1
        ).mean()
        
        # Pharmacophore feature recovery
        pharm_accuracy = (
            (pred_frags['pharm'] > 0.5) == 
            (true_frags[:, 167:167+27] > 0.5)
        ).float().mean()
        
        return {
            'maccs_similarity': maccs_similarity,
            'pharmacophore_accuracy': pharm_accuracy
        }

    def _compute_representation_metrics(self, pred_atoms, pred_frags):
        """Assess representation quality"""
        # Calculate entropy for each atom feature type
        atom_entropies = {}
        for feat_name, pred_feat in pred_atoms.items():
            feat_probs = F.softmax(pred_feat, dim=1)
            entropy = -torch.sum(feat_probs * torch.log(feat_probs + 1e-10), dim=1).mean()
            atom_entropies[f'{feat_name}_entropy'] = entropy

        # Calculate entropy for fragment features
        frag_entropy = -torch.sum(
            torch.sigmoid(pred_frags['maccs']) * 
            torch.log(torch.sigmoid(pred_frags['maccs']) + 1e-10)
        ).mean()

        metrics = {
            **atom_entropies,
            'fragment_entropy': frag_entropy
        }
        
        return metrics