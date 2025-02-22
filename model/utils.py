from matplotlib import pyplot as plt
from torch import nn
import numpy as np
import torch
from sklearn.metrics import auc, f1_score, matthews_corrcoef, mean_absolute_error, mean_squared_error, precision_recall_curve, precision_score, r2_score, recall_score,\
    roc_auc_score, accuracy_score, log_loss, confusion_matrix, roc_curve

def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

def compute_need_metrics(pred, target):
    # 将prob转换为预测标签
    pred_labels = (pred >= 0.5).int().cpu()
    target_labels = target.int().cpu()
    
    # 计算accuracy
    accuracy = accuracy_score(target_labels, pred_labels)

    # 计算precision, recall (sensitivity), specificity, F1 score
    precision = precision_score(target_labels, pred_labels, average='binary')
    sensitivity = recall_score(target_labels, pred_labels, average='binary')  # sensitivity is recall
    specificity = recall_score(target_labels, pred_labels, pos_label=0, average='binary')
    f1 = f1_score(target_labels, pred_labels, average='binary')
    
    # 计算MCC
    mcc = matthews_corrcoef(target_labels, pred_labels)

    return accuracy, precision, sensitivity, specificity, f1, mcc

def compute_confusion_matrix(all_preds, all_trues, threshold=0.5):
    all_preds = (all_preds >= threshold).int()
    all_trues = all_trues.int()

    cm = confusion_matrix(all_trues, all_preds)

    return cm

def plot_roc_curve(all_preds, all_trues, save_path):
    # 将预测的概率值和真实标签转换为 numpy 数组
    all_preds = all_preds.numpy()
    all_trues = all_trues.numpy()

    # 计算 FPR 和 TPR
    fpr, tpr, _ = roc_curve(all_trues, all_preds)
    roc_auc = auc(fpr, tpr)

    # 绘制 ROC 曲线
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'{save_path}/roc_curve.png', dpi=300)

    return roc_auc

def remove_nan_label(pred,truth):
    nan = torch.isnan(truth)
    truth = truth[~nan]
    pred = pred[~nan]

    return pred,truth

def roc_auc(pred,truth):
    return roc_auc_score(truth,pred)

def rmse(pred,truth):
    return nn.functional.mse_loss(pred,truth)**0.5

def mae(pred,truth):
    return mean_absolute_error(truth,pred)

func_dict={'relu':nn.ReLU(),
           'sigmoid':nn.Sigmoid(),
           'tanh':nn.Tanh(),
           'gelu':nn.GELU(),
           'leakyrelu':nn.LeakyReLU(),
           'mse':nn.MSELoss(),
           'rmse':rmse,
           'mae':mae,
           'crossentropy':nn.CrossEntropyLoss(),
           'bce':nn.BCEWithLogitsLoss(),
           'auc':roc_auc,
           }

def get_func(fn_name):
    # check if fn_name is already a function
    if callable(fn_name):
        return fn_name
    fn_name = fn_name.lower()
    return func_dict[fn_name]
