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
        info("Confusion Matrix:")
        info(cm)

        roc_auc = plot_roc_curve(all_preds, all_trues, save_path=self.cfg.logger.log_dir)
        info("AUC-ROC:", roc_auc)

        metrics_tracker['accuracy'] = accuracy
        metrics_tracker['precision'] = precision
        metrics_tracker['sensitivity'] = sensitivity
        metrics_tracker['specificity'] = specificity
        metrics_tracker['f1'] = f1
        metrics_tracker['mcc'] = mcc

        info(metrics_tracker)