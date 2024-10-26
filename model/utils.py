from torch import nn
import numpy as np
import torch
from sklearn.metrics import auc, f1_score, matthews_corrcoef, mean_absolute_error, mean_squared_error, precision_recall_curve, precision_score, r2_score, recall_score,\
    roc_auc_score, accuracy_score, log_loss

def compute_accuracy(pred, target):
    return float(torch.sum(torch.max(pred.detach(), dim = 1)[1] == target).cpu().item())/len(pred)

def compute_need_metrics(pred, target):
    # 将prob转换为预测标签
    pred_labels = (pred >= 0.5).int().cpu()
    target_labels = target.int().cpu()
    
    # 计算accuracy
    accuracy = (pred_labels == target_labels).float().mean().item()

    # 计算precision, recall (sensitivity), specificity, F1 score
    precision = precision_score(target_labels, pred_labels, average='binary')
    sensitivity = recall_score(target_labels, pred_labels, average='binary')  # sensitivity is recall
    specificity = recall_score(target_labels, pred_labels, pos_label=0, average='binary')
    f1 = f1_score(target_labels, pred_labels, average='binary')
    
    # 计算MCC
    mcc = matthews_corrcoef(target_labels, pred_labels)

    return accuracy, precision, sensitivity, specificity, f1, mcc

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
    fn_name = fn_name.lower()
    return func_dict[fn_name]
