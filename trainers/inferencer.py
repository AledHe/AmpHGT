import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.std_logger import info
from model.utils import compute_need_metrics, compute_confusion_matrix, plot_roc_curve
from collections import defaultdict
from sklearn.metrics import silhouette_score
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from umap import UMAP

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

    def umap(self, n_neighbors=15, min_dist=0.1, out_path=None):
        self.model.eval()
        features_dict = {}
        labels_list = []
        with torch.no_grad():
            for step, (batch, label) in enumerate(self.dataloaders["test"]):
                batch = batch.to(self.device)
                label = label.to(self.device)

                # Get intermediate features from all stages as a dict.
                logits, intermediate = self.model(batch, return_intermediate=True)

                # Initialize keys on the first iteration.
                if step == 0:
                    for stage_key, feats in intermediate.items():
                        features_dict[stage_key] = [feats.cpu()]
                else:
                    for stage_key, feats in intermediate.items():
                        features_dict[stage_key].append(feats.cpu())

                labels_list.append(label.cpu())

        # Combine all labels.
        all_labels = torch.cat(labels_list, dim=0).numpy()

        # For each feature stage, perform UMAP projection.
        for stage_key, feats_list in features_dict.items():
            all_features = torch.cat(feats_list, dim=0).numpy()

            info(f"Performing UMAP projection for : {stage_key}, shape={all_features.shape}")

            scaler = MinMaxScaler()
            all_features_std = scaler.fit_transform(all_features)

            reducer = UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=3,
                random_state=42,
                metric='cosine',
            )
            proj_3d = reducer.fit_transform(all_features_std)

            pos_idx = (all_labels == 1)
            neg_idx = (all_labels == 0)

            # 创建包含两个子图的画布
            fig = plt.figure(figsize=(16, 6))  # 横向排列两个子图
            
            # ---------------------------
            # 第一个子图：视角1 (elev=30, azim=45)
            # ---------------------------
            ax1 = fig.add_subplot(121, projection='3d')  # 1行2列的第1个子图
            ax1.scatter(
                proj_3d[neg_idx, 0], proj_3d[neg_idx, 1], proj_3d[neg_idx, 2],
                c='blue', alpha=0.5, label='Class 0', depthshade=True
            )
            ax1.scatter(
                proj_3d[pos_idx, 0], proj_3d[pos_idx, 1], proj_3d[pos_idx, 2],
                c='red', alpha=0.5, label='Class 1', depthshade=True
            )
            ax1.view_init(elev=30, azim=45)  # 设置视角
            ax1.set_title("View: elev=30°, azim=45°")
            ax1.legend()
            
            # ---------------------------
            # 第二个子图：视角2 (elev=-150, azim=45)
            # ---------------------------
            ax2 = fig.add_subplot(122, projection='3d')  # 1行2列的第2个子图
            ax2.scatter(
                proj_3d[neg_idx, 0], proj_3d[neg_idx, 1], proj_3d[neg_idx, 2],
                c='blue', alpha=0.5, label='Class 0', depthshade=True
            )
            ax2.scatter(
                proj_3d[pos_idx, 0], proj_3d[pos_idx, 1], proj_3d[pos_idx, 2],
                c='red', alpha=0.5, label='Class 1', depthshade=True
            )
            ax2.view_init(elev=30, azim=225)  # 对称视角
            ax2.set_title("View: elev=30°, azim=225°")
            ax2.legend()
            # 全局标题和保存
            plt.suptitle(
                f"3D UMAP ({stage_key}) - Two Views\n"
                f"n_neighbors={n_neighbors}, min_dist={min_dist}",
                y=1.0, fontsize=12
            )
            plt.tight_layout()
            
            stage_out_path = f"{out_path}/3d_two_views_{stage_key}_neib_{n_neighbors}_dist_{min_dist}.png"
            plt.savefig(stage_out_path, dpi=50, bbox_inches='tight')
            plt.close()