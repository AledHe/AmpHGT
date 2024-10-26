"""
Trainer adapted new pretrain heads.
"""

import json
import os
from typing import Dict

import mlflow
import torch

# from model.pretrain_heads import ReconstructionLoss
from model.PepMAE import PepMAELoss
from utils.std_logger import info
from collections import defaultdict
from pathlib import Path

class Masking_Trainer(object):
    def __init__(self,
                 cfg, 
                 dataloaders,
                 encoder,  # PharmHGT
                 decoder,  # PepMAE
                 optimizer,
                 device,
                 outputdir,
                 reconstruction_weight=1.0,
                 classification_weight=1.0
                 ):
        self.cfg = cfg
        self.dataloaders = dataloaders
        self.device = device
        self.outputdir = outputdir

        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.criterion = PepMAELoss(
            reconstruction_weight=reconstruction_weight,
            classification_weight=classification_weight
        ).to(device)

        # Training state
        self.epoch = 0
        self.global_train_step = 0
        # self.best_loss = float('inf')
        self.patience_counter = 0
        self.monitor_steps = cfg.train.get('log_interval', 10)

        # metrics tracking
        self.best_metrics = {
            'reconstruction_loss': float('inf'),
            'classification_loss': float('inf'),
            'total_loss': float('inf')
        }

    def _prepare_batch_targets(self, batch, atom_mask, fragment_mask):
        """
        Prepare both reconstruction and classification targets
        Now including masks to properly filter the targets
        """
        reconstruction_targets = {
            'atom': {
                'atom_features': batch.nodes['a'].data['f'][atom_mask],  # Only masked atoms
                'junction_features': batch.nodes['a'].data['f_junc'][atom_mask]
            },
            'fragment': {
                'fragment_features': batch.nodes['p'].data['f'][fragment_mask],  # Only masked fragments
                'junction_features': batch.nodes['p'].data['f_junc'][fragment_mask],
                'maccs': batch.nodes['p'].data['f'][fragment_mask, :167],
                'pharm': batch.nodes['p'].data['f'][fragment_mask, 167:167+27]
            }
        }

        classification_targets = {
            'atom': {
                'atomic_num': batch.nodes['a'].data['label'][atom_mask, 0],
            },
            'fragment': {
                'vocab_idx': batch.nodes['p'].data['label'][fragment_mask]
            }
        }

        return reconstruction_targets, classification_targets

    def _log_training_progress(self, metrics: Dict[str, float], step: int) -> str:
        """Format metrics for logging"""
        log_str = f"Step {step} | "
        log_str += f"Total Loss: {metrics['total_loss']:.4f} | "
        log_str += f"Recon Loss: {metrics['reconstruction_loss']:.4f} | "
        log_str += f"Class Loss: {metrics['classification_loss']:.4f} | "
        
        # Add accuracies
        for key, value in metrics.items():
            if 'accuracy' in key:
                log_str += f"{key}: {value:.2f} | "
        
        info(log_str)

    def train_epoch(self):
        """Single training epoch with PepMAE"""
        self.encoder.train()
        self.decoder.train()
        
        metrics_tracker = defaultdict(float)
        num_batches = 0

        for step, (batch, label) in enumerate(self.dataloaders['train']):
            batch = batch.to(self.device)

            # Get masks
            atom_mask = batch.nodes['a'].data['mask']
            fragment_mask = batch.nodes['p'].data['mask']

            # Prepare targets with masks
            reconstruction_targets, classification_targets = self._prepare_batch_targets(
                batch, atom_mask, fragment_mask
            )

            # Forward pass through encoder (PharmHGT)
            atom_repr, fragment_repr = self.encoder(batch)

            # Forward pass through decoder (PepMAE)
            reconstruction_pred, classification_pred = self.decoder(
                atom_repr, fragment_repr,
                atom_mask, fragment_mask
            )

            # Compute loss
            loss, batch_metrics = self.criterion(
                reconstruction_pred, reconstruction_targets,
                classification_pred, classification_targets
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            for name, value in batch_metrics.items():
                metrics_tracker[name] += value
            num_batches += 1

            # Log progress
            self.global_train_step += 1
            if self.global_train_step % self.monitor_steps == 0:
                current_metrics = {k: v/num_batches for k, v in metrics_tracker.items()}
                self._log_training_progress(current_metrics, step)

        return {k: v/num_batches for k, v in metrics_tracker.items()}

    def validate_epoch(self):
        """Validation epoch"""
        pass

    def run(self):
        """Complete training loop"""
        for epoch in range(self.cfg.train.epochs):
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()

            # Logging
            self._log_epoch_metrics(train_metrics, val_metrics, epoch)

            # Model selection and early stopping
            if val_metrics['total_loss'] < self.best_loss:
                self.best_loss = val_metrics['total_loss']
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg.train.patience:
                    info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

    def _log_epoch_metrics(self, train_metrics, val_metrics, epoch):
        """Log metrics for both training and validation"""
        info(f"\nEpoch {epoch + 1}")
        info("Training Metrics:")
        for name, value in train_metrics.items():
            info(f"{name}: {value:.4f}")
            if self.cfg.logger.log:
                mlflow.log_metric(f"train_{name}", value, step=epoch)

        info("\nValidation Metrics:")
        for name, value in val_metrics.items():
            info(f"{name}: {value:.4f}")
            if self.cfg.logger.log:
                mlflow.log_metric(f"val_{name}", value, step=epoch)

    def _save_checkpoint(self, epoch, metrics):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.outputdir, f"checkpoint_epoch_{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }, os.path.join(checkpoint_dir, "model.pt"))
        
        with open(os.path.join(checkpoint_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f)