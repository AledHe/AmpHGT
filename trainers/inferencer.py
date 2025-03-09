import numpy as np
import torch
from utils.std_logger import info
from model.utils import compute_need_metrics, compute_confusion_matrix, plot_roc_curve
from collections import defaultdict

class Inferencer_Binary(object):
    def __init__(self,
                 cfg, 
                 dataloaders,
                 model,  # AmpHGT_FT
                 device,
                 ):
        self.cfg = cfg
        self.dataloaders = dataloaders
        self.device = device

        self.model = model

    def infer(self):
        # model.eval is already enabled in loading model.
        metrics_tracker = defaultdict(float)

        all_preds = []
        all_trues = []

        with torch.no_grad():
            for step, (batch, label) in enumerate(self.dataloaders["test"]):
                batch = batch.to(self.device) # dgl.batch(list(graphs))
                label = label.to(self.device) # torch.tensor(list(labels), dtype=torch.float32)

                logits = self.model(batch)

                logits = torch.sigmoid(logits) # convert logits to probabilities

                all_preds.append(logits.cpu())
                all_trues.append(label.cpu())

        all_preds = torch.cat(all_preds, dim=0)
        all_trues = torch.cat(all_trues, dim=0)

        accuracy, precision, sensitivity, specificity, f1, mcc = compute_need_metrics(all_preds, all_trues)
        cm = compute_confusion_matrix(all_preds, all_trues)

        roc_auc = plot_roc_curve(all_preds, all_trues, save_path=self.cfg.logger.log_dir)

        metrics_tracker['accuracy'] = accuracy
        metrics_tracker['precision'] = precision
        metrics_tracker['sensitivity'] = sensitivity
        metrics_tracker['specificity'] = specificity
        metrics_tracker['f1'] = f1
        metrics_tracker['mcc'] = mcc
        metrics_tracker['roc_auc'] = roc_auc

        metric_labels = {
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'sensitivity': 'Sensitivity',
            'specificity': 'Specificity',
            'f1': 'F1 Score',
            'mcc': 'MCC',
            'roc_auc': 'ROC AUC'
        }
        max_len = max(len(label) for label in metric_labels.values())
        header = f"\n{' Evaluation Metrics ':=^{max_len+14}}"
        footer = f"{'':=<{max_len+14}}"
        
        metrics_str = [header]
        for key in ['accuracy', 'precision', 'sensitivity', 'specificity', 'f1', 'mcc', 'roc_auc']:
            label = metric_labels[key]
            value = metrics_tracker[key]
            metrics_str.append(f"| {label:<{max_len}} | {value:^8.4f} |")
        
        metrics_str.append(footer)
        info('\n'.join(metrics_str))

        cm_array = np.array(cm)
        total = cm_array.sum()
        header = "\nConfusion Matrix (Counts)"
        table_str = [
            header,
            "          Actual",
            "        Neg    Pos",
            f"Pred Neg  {cm[0][0]:<4}  {cm[0][1]:<4}",
            f"     Pos  {cm[1][0]:<4}  {cm[1][1]:<4}",
            f"\nTotal samples: {total}"
        ]
        info('\n'.join(table_str))