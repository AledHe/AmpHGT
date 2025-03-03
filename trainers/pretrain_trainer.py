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
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.monitor_steps = cfg.train.get('log_interval', 10)

        self.early_stop = False
        self.val_interval = cfg.train.val_interval

    def _log_progress(self, metrics: Dict[str, float], step: int, stage: str) -> str:
        """Format metrics for logging"""
        log_str = f"{stage} Step {step} | "
        log_str += f"Total Loss: {metrics['loss']:.4f} | "
        log_str += f"Recon Loss: {metrics['recon_loss']:.4f} | "
        log_str += f"Class Loss: {metrics['class_loss']:.4f} | "
        
        # Add accuracies
        for key, value in metrics.items():
            if 'acc' in key:
                log_str += f"{key}: {value:.2f} | "
        
        info(log_str)

    def prepare_mask_targets(self, batch, atom_repr, fragment_repr, residue_repr):
        # get masked batch
        atom_mask = atom_repr[batch.nodes['a'].data['mask'] == True]
        fragment_mask = fragment_repr[batch.nodes['p'].data['mask'] == True]
        residue_mask = residue_repr[batch.nodes['rsd'].data['mask'] == True]
        # get target
        feature_targets = {
            "atom": batch.nodes['a'].data['f_unmasked'][batch.nodes['a'].data['mask'] == True],
            "fragment": batch.nodes['p'].data['f_unmasked'][batch.nodes['p'].data['mask'] == True]
        }
        class_targets = {
            "atom": batch.nodes['a'].data['label'][batch.nodes['a'].data['mask'] == True],
            "fragment": batch.nodes['p'].data['label'][batch.nodes['p'].data['mask'] == True],
            "residue": batch.nodes['rsd'].data['label_orig'][batch.nodes['rsd'].data['mask'] == True]
        }

        return atom_mask, fragment_mask, residue_mask, feature_targets, class_targets

    def train_epoch(self):
        """Single training epoch with PepMAE"""
        self.encoder.train()
        self.decoder.train()
        
        epoch_metrics = defaultdict(float)
        interval_metrics = defaultdict(float)
        interval_samples = 0

        num_batches = 0

        for step, (batch, label) in enumerate(self.dataloaders['train']):
            if self.early_stop:
                # stop training if early stop triggered
                break
            # print(step, (batch, label))

            batch = batch.to(self.device)
            label = label.to(self.device) # -1, 0, 1

            # Forward pass through encoder (PharmHGT), returns stacked _h and _junc_h.
            atom_repr, fragment_repr, rsd_repr, bg, cls_emb = self.encoder(batch)
            batch = bg

            # prepare mask and targets for MLP decoder. other decoder won't use this, return and pass for unified interface.
            # while the targets are used for all encoders.
            atom_mask, fragment_mask, residue_mask, feature_targets, class_targets = self.prepare_mask_targets(
                batch, atom_repr, fragment_repr, rsd_repr
            )

            # Forward pass through decoder (PepMAE)
            feature_decodes, class_decodes = self.decoder(
                batch, atom_mask, fragment_mask, residue_mask
            )

            # Compute loss
            # batch_metrics = {
            #     "loss": total_loss.item(),
            #     "recon_loss": recon_loss.item(),
            #     "class_loss": class_loss.item(),
            #     "atom_acc": atom_acc,
            #     "fragment_acc": frag_acc
            # }
            loss, batch_metrics = self.criterion(
                feature_decodes, feature_targets,
                class_decodes, class_targets
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            for name, value in batch_metrics.items():
                epoch_metrics[name] += value
                interval_metrics[name] += value
            num_batches += 1
            interval_samples += 1
            
            # Log progress
            if self.global_train_step % self.monitor_steps == 0:
                current_metrics = {
                    k: v/interval_samples
                    for k, v in interval_metrics.items()
                }
                self._log_progress(current_metrics, step, "train")
                
                interval_metrics.clear()
                interval_samples = 0

            # -----------------------------
            # every val_interval steps, run validation
            if self.global_train_step % self.val_interval == 0:
                self.in_train_validate()
                # if early stop
                if self.early_stop:
                    break

        return {k: v/num_batches for k, v in epoch_metrics.items()}
    
    def in_train_validate(self):
        """
        触发一次验证，得到 val_metrics 并与 best_loss 比较，若更好则保存模型。
        如果 patience 超出限制，则 early_stop = True.
        """
        val_metrics = self.eov_epoch('valid')
        
        # log validation metrics
        if self.cfg.logger.log:
            for name, value in val_metrics.items():
                mlflow.log_metric(f"val_{name}", value, step=self.global_train_step)

        # early-stopping & checkpoint
        if val_metrics['loss'] < self.best_loss:
            self.best_loss = val_metrics['loss']
            self.patience_counter = 0
            self._save_checkpoint(
                self.epoch,
                val_metrics,
                step_suffix=f"step_{self.global_train_step}"
            )
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.cfg.train.patience:
                info(f"Early stopping triggered at step {self.global_train_step}.")
                self.early_stop = True

    def eov_epoch(self, stage: str):
        """evaluation or validation epoch"""
        self.encoder.eval()
        self.decoder.eval()
        
        metrics_tracker = defaultdict(float)
        num_batches = 0

        with torch.no_grad():
            for step, (batch, label) in enumerate(self.dataloaders[stage]):
                batch = batch.to(self.device)
                label = label.to(self.device)
                
                # Forward pass through encoder (PharmHGT)
                atom_repr, fragment_repr, rsd_repr, bg, cls_emb = self.encoder(batch)
                batch = bg
                
                atom_mask, fragment_mask, residue_mask, feature_targets, class_targets = self.prepare_mask_targets(
                    batch, atom_repr, fragment_repr, rsd_repr
                )

                # Forward pass through decoder (PepMAE)
                feature_decodes, class_decodes = self.decoder(
                    batch, atom_mask, fragment_mask, residue_mask
                )

                # Compute loss
                loss, batch_metrics = self.criterion(
                    feature_decodes, feature_targets,
                    class_decodes, class_targets
                )

                # Update metrics
                for name, value in batch_metrics.items():
                    metrics_tracker[name] += value
                num_batches += 1

        if num_batches > 0:
            for name in metrics_tracker:
                metrics_tracker[name] /= num_batches
        
        return metrics_tracker

    def run(self):
        """Complete training loop"""
        for epoch in range(self.cfg.train.epochs):
            if self.early_stop:
                break

            # training
            train_metrics = self.train_epoch()

            self._log_epoch_metrics(train_metrics=train_metrics, val_metrics=None, test_metrics=None, epoch=epoch)

            if self.early_stop:
                break
            
            # validation
            val_metrics = self.eov_epoch(stage='valid')

            self._log_epoch_metrics(train_metrics=None, val_metrics=val_metrics, test_metrics=None, epoch=epoch)

            # Model selection and early stopping
            if val_metrics['loss'] < self.best_loss:
                self.best_loss = val_metrics['loss']
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics, "best")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg.train.patience:
                    info(f"Early stopping triggered after epoch {epoch + 1}.")
                    break
        
        # test
        test_metrics = self.eov_epoch(stage='test')
        self._log_epoch_metrics(train_metrics=None, val_metrics=None, test_metrics=test_metrics, epoch=epoch)

    def _log_epoch_metrics(self, train_metrics, val_metrics, test_metrics, epoch):
        """Log metrics for both training and validation"""
        info(f"Epoch {epoch + 1}")
        if train_metrics:
            info("Training Metrics:")
            for name, value in train_metrics.items():
                info(f"{name}: {value:.4f}")
                if self.cfg.logger.log:
                    mlflow.log_metric(f"train_{name}", value, step=epoch)
        elif val_metrics:
            info("Validation Metrics:")
            for name, value in val_metrics.items():
                info(f"{name}: {value:.4f}")
                if self.cfg.logger.log:
                    mlflow.log_metric(f"val_{name}", value, step=epoch)
        elif test_metrics:
            info("Test Metrics:")
            for name, value in test_metrics.items():
                info(f"{name}: {value:.4f}")
                if self.cfg.logger.log:
                    mlflow.log_metric(f"test_{name}", value, step=epoch)

    def _save_checkpoint(self, epoch, metrics, step_suffix=""):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(
            self.outputdir, 
            f"checkpoint_epoch_{epoch}" + (f"_{step_suffix}" if step_suffix else "")
        )
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

        info(f"Checkpoint saved at {checkpoint_dir}.")