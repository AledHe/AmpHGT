"""
Trainer for finetune
"""

import json
import os

import mlflow
import torch
import torch.nn as nn

from typing import Dict

from utils.std_logger import info
from model.utils import compute_need_metrics
from collections import defaultdict

class Finetune_Trainer(object):
    def __init__(self,
                 cfg, 
                 dataloaders,
                 model,  # PharmHGT_FP
                 optimizer,
                 scheduler,
                 device,
                 outputdir,
                 ):
        self.cfg = cfg
        self.dataloaders = dataloaders
        self.device = device
        self.outputdir = outputdir

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.criterion = nn.BCEWithLogitsLoss().to(device) # for binary classification

        # Training state
        self.epoch = 0
        self.global_train_step = 0
        self.best_mcc = -1.0
        self.patience_counter = 0
        self.monitor_steps = cfg.train.get('log_interval', 10)

        self.early_stop = False

    def _log_progress(self, metrics: Dict[str, float], step: int, stage: str) -> str:
        """Format metrics for logging"""
        log_str = f"{stage} Step {step}, lr {self.scheduler.get_lr()} | "
        log_str += f"Loss: {metrics['loss']:.4f} | "
        
        info(log_str)

    def train_epoch(self):
        self.model.train()

        epoch_metrics = defaultdict(float)
        interval_metrics = defaultdict(float)
        interval_samples = 0

        num_batches = 0

        for step, (batch, label) in enumerate(self.dataloaders['train']):
            if self.early_stop:
                # stop training if early stop triggered
                break
            # print(step, (batch, label))

            batch = batch.to(self.device) # dgl.batch(list(graphs))
            label = label.to(self.device) # torch.tensor(list(labels), dtype=torch.float32)

            # Forward pass through encoder (PharmHGT_FP with pooling), returns bg_embeds.
            logits = self.model(batch)

            # Compute loss
            loss = self.criterion(logits, label.unsqueeze(1))  # shape [batch_size]

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            epoch_metrics['loss'] += loss.item()
            interval_metrics['loss'] += loss.item()
            num_batches += 1
            interval_samples += 1

            # Log progress
            self.global_train_step += 1
            if self.global_train_step % self.monitor_steps == 0:
                avg_loss = interval_metrics['loss']/num_batches
                self._log_progress({'loss': avg_loss}, step, "train")

                interval_metrics.clear()
                interval_samples = 0

        return {k: v/num_batches for k, v in epoch_metrics.items()}
    
    def eov_epoch(self, stage: str):
        """evaluation or validation epoch"""
        self.model.eval()

        metrics_tracker = defaultdict(float)
        num_batches = 0
        total_loss = 0

        all_preds = []
        all_trues = []

        with torch.no_grad():
            for step, (batch, label) in enumerate(self.dataloaders[stage]):
                batch = batch.to(self.device) # dgl.batch(list(graphs))
                label = label.to(self.device) # torch.tensor(list(labels), dtype=torch.float32)

                # Forward pass through encoder (PharmHGT_FP with pooling), returns bg_embeds.
                logits = self.model(batch)

                # Compute loss
                loss = self.criterion(logits, label.unsqueeze(1))
                total_loss += loss.item()

                logits = torch.sigmoid(logits) # convert logits to probabilities

                all_preds.append(logits.cpu())
                all_trues.append(label.cpu())

                num_batches += 1

        all_preds = torch.cat(all_preds, dim=0)
        all_trues = torch.cat(all_trues, dim=0)

        accuracy, precision, sensitivity, specificity, f1, mcc = compute_need_metrics(all_preds, all_trues)

        average_loss = total_loss / num_batches

        metrics_tracker['loss'] = average_loss
        metrics_tracker['accuracy'] = accuracy
        metrics_tracker['precision'] = precision
        metrics_tracker['sensitivity'] = sensitivity
        metrics_tracker['specificity'] = specificity
        metrics_tracker['f1'] = f1
        metrics_tracker['mcc'] = mcc

        return metrics_tracker
    
    def run(self):
        best_checkpoint_path = None

        for epoch in range(self.cfg.train.epochs):
            if self.early_stop:
                break

            train_metrics = self.train_epoch()

            self._log_epoch_metrics(train_metrics=train_metrics, val_metrics=None, test_metrics=None, epoch=epoch)

            if self.early_stop:
                break

            val_metrics = self.eov_epoch("valid")

            self._log_epoch_metrics(train_metrics=None, val_metrics=val_metrics, test_metrics=None, epoch=epoch)

            # Model selection and early stopping
            if val_metrics['mcc'] > self.best_mcc:
                self.best_mcc = val_metrics['mcc']
                self.patience_counter = 0
                checkpoint_path = self._save_checkpoint(epoch, val_metrics, "best")
                best_checkpoint_path = checkpoint_path
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg.train.patience:
                    info(f"Early stopping triggered after epoch {epoch + 1}.")
                    break

        if best_checkpoint_path:
            info(f"Loading best model from {best_checkpoint_path} for testing")
            checkpoint = torch.load(
                os.path.join(best_checkpoint_path, "model.pt"),
                map_location=self.device
            )
            
            state_dict = {
                k.replace('module.', ''): v 
                for k, v in checkpoint['model_state_dict'].items()
            }
            
            self.model.load_state_dict(state_dict, strict=False)
            
            if self.cfg.train.sq_embed == "ESM2":
                if hasattr(self.model.pretrain_model, 'rsd_encoder'):
                    self.model.pretrain_model.rsd_encoder.eval()
            elif self.cfg.train.sq_embed == "Ensemble":
                if hasattr(self.model.pretrain_model, 'rsd_encoder') and \
                hasattr(self.model.pretrain_model.rsd_encoder, 'esm_encoder'):
                    self.model.pretrain_model.rsd_encoder.esm_encoder.eval()
            
            self.model.eval()

        # test the best model
        test_metrics = self.eov_epoch("test")
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
        
        if self.cfg.train.sq_embed == "ESM2":
            model_state_dict = {
                k: v for k, v in self.model.state_dict().items()
                if not k.startswith('pretrain_model.rsd_encoder')
            }
        elif self.cfg.train.sq_embed == "Ensemble":
            model_state_dict = {
                k: v for k, v in self.model.state_dict().items()
                if not k.startswith('pretrain_model.rsd_encoder.esm_encoder')
            }
        else:
            model_state_dict = {
                k: v for k, v in self.model.state_dict().items()
            }
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
        }, os.path.join(checkpoint_dir, "model.pt"))
        
        with open(os.path.join(checkpoint_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f)

        info(f"Checkpoint saved at {checkpoint_dir}.")

        return checkpoint_dir

class Finetune_Trainer_RG(object):
    def __init__(self,
                 cfg, 
                 dataloaders,
                 model,  # PharmHGT_FP
                 optimizer,
                 device,
                 outputdir,
                 ):
        self.cfg = cfg
        self.dataloaders = dataloaders
        self.device = device
        self.outputdir = outputdir

        self.model = model
        self.optimizer = optimizer

        self.criterion = nn.MSELoss().to(device) # for regression
        # use it in torch.sqrt(mse_loss) for RMSE

        # Training state
        self.epoch = 0
        self.global_train_step = 0
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.monitor_steps = cfg.train.get('log_interval', 10)

        self.early_stop = False