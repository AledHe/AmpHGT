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
                 outputdir):
        self.cfg = cfg
        self.dataloaders = dataloaders
        self.device = device
        self.outputdir = outputdir

        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.criterion = PepMAELoss().to(device)

        # Training state
        self.epoch = 0
        self.global_train_step = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.monitor_steps = cfg.train.get('log_interval', 10)

    def _prepare_batch_features(self, batch):
        """Prepare original features before masking"""
        return {
            'atom_features': {
                'atom_features': batch.nodes['a'].data['f'].clone(),
                'junction_features': batch.nodes['a'].data['f_junc'].clone()
            },
            'fragment_features': {
                'fragment_features': batch.nodes['p'].data['f'].clone(),
                'junction_features': batch.nodes['p'].data['f_junc'].clone(),
                'maccs': batch.nodes['p'].data['f'][:, :167].clone(),  # MACCS keys
                'pharm': batch.nodes['p'].data['f'][:, 167:167+27].clone()  # Pharmacophore features
            }
        }

    def _format_metrics(self, metrics: Dict[str, float], step: int) -> str:
        """Format metrics for logging"""
        atom_metrics = {k: v for k, v in metrics.items() if 'atom' in k}
        frag_metrics = {k: v for k, v in metrics.items() if 'fragment' in k}
        
        avg_atom_loss = sum(v for k, v in atom_metrics.items() if 'loss' in k) / len(atom_metrics)
        avg_frag_loss = sum(v for k, v in frag_metrics.items() if 'loss' in k) / len(frag_metrics)
        
        return (f"Step {step} | "
                f"Total Loss: {metrics['total_loss']:.4f} | "
                f"Atom Loss: {avg_atom_loss:.4f} | "
                f"Fragment Loss: {avg_frag_loss:.4f}")

    def train_epoch(self):
        """Single training epoch with PepMAE"""
        self.encoder.train()
        self.decoder.train()
        
        total_loss = 0
        batch_metrics = defaultdict(float)
        num_batches = 0

        for step, (batch, _) in enumerate(self.dataloaders['train']):
            batch = batch.to(self.device)

            # Store original features and masks
            original_features = self._prepare_batch_features(batch)
            atom_mask = batch.nodes['a'].data['mask']
            fragment_mask = batch.nodes['p'].data['mask']

            # Forward pass through encoder (PharmHGT)
            atom_repr, fragment_repr = self.encoder(batch)

            # Forward pass through decoder (PepMAE)
            atom_pred, fragment_pred = self.decoder(
                atom_repr, fragment_repr,
                atom_mask, fragment_mask
            )

            # Compute loss
            loss, loss_metrics = self.criterion(
                atom_pred,
                {k: v[atom_mask] for k, v in original_features['atom_features'].items()},
                fragment_pred,
                {k: v[fragment_mask] for k, v in original_features['fragment_features'].items()}
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            for k, v in loss_metrics.items():
                batch_metrics[k] += v
            num_batches += 1

            # Log progress
            self.global_train_step += 1
            if self.global_train_step % self.monitor_steps == 0:
                current_metrics = {k: v/num_batches for k, v in batch_metrics.items()}
                info(self._format_metrics(current_metrics, self.global_train_step))

        # Compute epoch metrics
        avg_metrics = {k: v/num_batches for k, v in batch_metrics.items()}
        avg_metrics['total_loss'] = total_loss/num_batches

        return avg_metrics

    def validate_epoch(self):
        """Validation epoch"""
        self.encoder.eval()
        self.decoder.eval()
        
        batch_metrics = defaultdict(float)
        num_batches = 0

        with torch.no_grad():
            for batch, _ in self.dataloaders['valid']:
                batch = batch.to(self.device)

                # Prepare features and masks
                original_features = self._prepare_batch_features(batch)
                atom_mask = batch.nodes['a'].data['mask']
                fragment_mask = batch.nodes['p'].data['mask']

                # Forward passes
                atom_repr, fragment_repr = self.encoder(batch)
                atom_pred, fragment_pred = self.decoder(
                    atom_repr, fragment_repr,
                    atom_mask, fragment_mask
                )

                # Compute metrics
                _, metrics = self.criterion(
                    atom_pred,
                    {k: v[atom_mask] for k, v in original_features['atom_features'].items()},
                    fragment_pred,
                    {k: v[fragment_mask] for k, v in original_features['fragment_features'].items()}
                )

                # Update batch metrics
                for k, v in metrics.items():
                    batch_metrics[k] += v
                num_batches += 1

        # Average metrics
        avg_metrics = {k: v/num_batches for k, v in batch_metrics.items()}
        return avg_metrics

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