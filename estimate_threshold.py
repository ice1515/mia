import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_curve, precision_score, recall_score, f1_score, \
    precision_recall_fscore_support, auc
from sklearn.metrics import roc_auc_score
from torch import nn
from MotifyEntropy import _Mentr




def estimate_threshold_with_shadow(shadowmodel, shadow_data, num_classes, metric="",mu=0.9):
    score = []
    ce_criterion = nn.CrossEntropyLoss()
    shadowmodel.eval()
    mem_groundtruth = np.ones(int(len(shadow_data) / 2))
    non_groundtruth = np.zeros(int(len(shadow_data) / 2))
    groundtruth = np.concatenate((mem_groundtruth, non_groundtruth))
    with torch.no_grad():
        for data in shadow_data:
            data = data.cuda()
            target = data.y.cpu()
            Soutput = shadowmodel(data.x, data.edge_index, data.batch)
            Sposterior = F.softmax(Soutput, dim=1).cpu()
            pred = np.argmax(Sposterior)
            # 获取阴影模型后验概率
            if metric == 'CELoss':
                # celoss = ce_criterion(Sposterior, target)
                celoss = ce_criterion(Sposterior, torch.LongTensor([pred]))
                score.append(-1 * celoss.item())
            elif metric == 'Maximum':
                score.append(torch.max(Sposterior))
            elif metric == 'ModifyEntropy':
                encoding_y = F.one_hot(target, num_classes=num_classes)
                score.extend(-1 *
                             _Mentr(np.asarray(Sposterior.detach().cpu()), np.asarray(encoding_y.detach().cpu())))

    score = np.asarray(score)
    if metric == 'ModifyEntropy':
        score_m, score_nm = score[:len(score) // 2], score[len(score) // 2:]
        shadow_data_m, shadow_data_nm = shadow_data[:len(shadow_data) // 2], shadow_data[len(shadow_data) // 2:]
        estimate_t = mem_inf_thre(num_classes, score_m, score_nm, shadow_data_m, shadow_data_nm,mu)
    else:
        estimate_t = determine_threshold(groundtruth, score, metric)

    return score, estimate_t


def determine_threshold(S_label_list, S_score_lst, metric):
    max_acc, max_acc_pre, max_acc_rec, max_acc_f1 = 0, 0, 0, 0
    max_acc_threshold = 0

    y_true = np.array(S_label_list)
    y_score = np.asarray(S_score_lst)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    AUC = auc(fpr, tpr)
    for thres in thresholds:
        y_pred = [1 if m >= thres else 0 for m in y_score]
        ev_acc = accuracy_score(y_true, y_pred)
        if ev_acc > max_acc:
            max_acc = ev_acc
            max_acc_pre = precision_score(y_true, y_pred, zero_division=0)
            max_acc_rec = recall_score(y_true, y_pred, zero_division=0)
            max_acc_f1 = f1_score(y_true, y_pred, zero_division=0)
            max_acc_threshold = thres
    print("Selection auc:%f,acc:%f, pre:%f, re:%f, f1:%f" % (AUC, max_acc, max_acc_pre, max_acc_rec, max_acc_f1))
    return max_acc_threshold


def cal_metric_threshold_mentr(y_true, y_score, targetdata, thres):
    from bisect import bisect_left
    y_true = np.array(y_true)

    y_pred = [1 if score >= thres[targetdata[idx].y.item()] else 0 for idx, score in enumerate(y_score)]
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    AUC = auc(fpr, tpr)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    fscore = f1_score(y_true, y_pred, zero_division=0)
    precision_marco, recall_marco, fscore_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    fp_list = [0.01, 0.1]
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

        print(
            't={},auc:{:.4f},acc:{:.4f},precision:{:0.4f},recall:{:.4f},f1:{:0.4f},TPR@FPR={:.2f}:{:.4f},'.format(thres,
                                                                                                                  AUC,
                                                                                                                  accuracy,
                                                                                                                  precision,
                                                                                                                  recall,
                                                                                                                  fscore,
                                                                                                                  fp_list[
                                                                                                                      idx],
                                                                                                                  low_fpr_tpr))
    print('precision_marco:{:.4f},recall_marco:{:.4f},f1score_macro:{:.4f}'.format(precision_marco, recall_marco,
                                                                                   fscore_macro))
    return [AUC, accuracy, precision, recall, fscore, low_fpr_tpr, precision_marco, recall_marco, fscore_macro, fpr,
            tpr]


def thre_setting(tr_values, te_values):
    value_list = np.concatenate((tr_values, te_values))
    thre, max_acc = 0, 0
    for value in value_list:
        tr_ratio = np.sum(tr_values >= value) / (len(tr_values) + 0.0)
        te_ratio = np.sum(te_values < value) / (len(te_values) + 0.0)
        acc = 0.5 * (tr_ratio + te_ratio)
        if acc > max_acc:
            thre, max_acc = value, acc
    return thre, max_acc


def mem_inf_thre(num_classes, s_tr_values, s_te_values, shadow_m, shadow_nm,mu):
    thres = []
    for num in range(num_classes):
        s_tr_values_y = [score for idx, score in enumerate(s_tr_values) if shadow_m[idx].y.item() == num]
        s_te_values_y = [score for idx, score in enumerate(s_te_values) if shadow_nm[idx].y.item() == num]

        thre, acc = thre_setting(s_tr_values_y, s_te_values_y)
        print('y={},thre:{:.4f},max_acc:{:4f}'.format(num, thre, acc))
        thres.append(thre)
    return thres


def cal_metric_threshold(y_true, y_score, t, metric=""):
    from bisect import bisect_left
    y_true = np.array(y_true)
    y_pred = [1 if m >= t else 0 for m in y_score]
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    AUC = auc(fpr, tpr)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    fscore = f1_score(y_true, y_pred, zero_division=0)
    precision_marco, recall_marco, fscore_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    fp_list = [0.01, 0.1]
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
    return [AUC, accuracy, precision, recall, fscore, low_fpr_tpr, precision_marco, recall_marco, fscore_macro, fpr,
            tpr]
