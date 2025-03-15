import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_curve, precision_score, recall_score, f1_score, \
    precision_recall_fscore_support, auc
from sklearn.metrics import roc_auc_score
from torch import nn
from MotifyEntropy import _Mentr


def estimate_threshold_with_shadow(shadowmodel, shadow_data, num_classes, metric='CELoss'):
    score = []
    mem_groundtruth = np.ones(int(len(shadow_data) / 2))
    non_groundtruth = np.zeros(int(len(shadow_data) / 2))
    groundtruth = np.concatenate((mem_groundtruth, non_groundtruth))
    ce_criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in shadow_data:
            data, target = data.cuda(), data.y.cuda()
            Soutput = shadowmodel(data.x, data.edge_index, data.batch)
            Sposterior = F.softmax(Soutput, dim=1)
            Slabel = Soutput.max(1)[1]
            if metric == 'CELoss':
                if Slabel != target:
                    score.append(100)
                else:
                    score.append(ce_criterion(Sposterior, target).item())
            elif metric == 'Maximum':
                score.append(torch.max(Sposterior).item())
            elif metric == 'NormalizedEntropy':
                entropy = -1 * torch.sum(torch.mul(Sposterior, torch.log(Sposterior)))
                if str(entropy.item()) == 'nan':  # 预处理
                    score.append(1e-100)
                else:
                    score.append(entropy.item())
            elif metric == 'ModifyEntropy':
                encoding_y = F.one_hot(target, num_classes=num_classes)
                score.extend(_Mentr(np.asarray(Sposterior.detach().cpu()), np.asarray(encoding_y.detach().cpu())))
            else:
                print('[!] Invalid type')

    score = np.asarray(score)
    estimate_t = determine_threshold(groundtruth, score, metric)
    return score, estimate_t


def determine_threshold(S_label_list, S_score_lst, metric):
    max_acc, max_acc_pre, max_acc_rec, max_acc_f1 = 0, 0, 0, 0
    max_acc_threshold = 0

    y_true = np.array(S_label_list)
    y_score = np.asarray(S_score_lst)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    for thres in thresholds:
        if metric in ['CELoss', 'NormalizedEntropy', 'ModifyEntropy']:
            y_pred = [1 if m < thres else 0 for m in y_score]
        elif metric in ['Maximum', 'gap']:
            y_pred = [1 if m > thres else 0 for m in y_score]
        ev_acc = accuracy_score(y_true, y_pred)
        if ev_acc > max_acc:
            max_acc = ev_acc
            max_acc_pre = precision_score(y_true, y_pred, zero_division=0)
            max_acc_rec = recall_score(y_true, y_pred, zero_division=0)
            max_acc_f1 = f1_score(y_true, y_pred, zero_division=0)
            max_acc_threshold = thres
    print("Selection acc:%f, pre:%f, re:%f, f1:%f" % (max_acc, max_acc_pre, max_acc_rec, max_acc_f1))
    return max_acc_threshold



def cal_metric_threshold(y_true, y_score, t, metric):
    from bisect import bisect_left
    y_true = np.array(y_true)

    y_pred = []
    if metric in ['CELoss', 'NormalizedEntropy', 'ModifyEntropy']:
        y_pred = [1 if m < t else 0 for m in y_score]
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=0)

    elif metric in ['Maximum', 'gap']:
        y_pred = [1 if m > t else 0 for m in y_score]
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    AUC = auc(fpr, tpr)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    fscore = f1_score(y_true, y_pred, zero_division=0)
    precision_marco, recall_marco, fscore_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    fp_list = [0.1]
    tp_list = []
    for idx, item in enumerate(fp_list):
        fp_index = bisect_left(fpr, item)
        if fp_index == 0:
            tp_list.append(tpr[0])
        elif fp_list == len(fpr):
            raise ValueError("Give fpr larger than 1.0")
        else:
            tp_list.append(tpr[fp_index - 1])
        low_fpr_tpr = tp_list[idx]

        print('t={},auc:{:.4f},acc:{:.4f},precision:{:0.4f},recall:{:.4f},f1:{:0.4f},TPR@FPR={}:{:.4f},'.format(t, AUC,
                                                                                                                accuracy,
                                                                                                                precision,
                                                                                                                recall,
                                                                                                                fscore,
                                                                                                                fp_list[
                                                                                                                    idx],
                                                                                                                low_fpr_tpr))
        print('precision_marco:{:.4f},recall_marco:{:.4f},f1score_macro:{:.4f}'.format(precision_marco, recall_marco,
                                                                                       fscore_macro))
    return [AUC, accuracy, precision, recall, fscore, low_fpr_tpr, precision_marco, recall_marco, fscore_macro]
