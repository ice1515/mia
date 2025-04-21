import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import roc_curve, \
    classification_report
from torch import nn

from MotifyEntropy import _Mentr
from estimate_threshold import estimate_threshold_with_shadow, cal_metric_threshold, cal_metric_threshold_mentr

''' code implemented from
    Reference1:"Membership Leakage in Label-Only Exposures" ""
    github1:https://github.com/zhenglisec/Label-Only-MIA/blob/main/attack.py

    Reference2:"Adapting Membership Inference Attacks to GNN for Graph Classification: Approaches and Implications"
    github2:https://github.com/TrustworthyGNN/MIA-GNN
'''


def plot_distribution(member, nonmember):
    plt.figure(figsize=(6, 5))
    print("Member data range:", np.min(member), np.max(member))
    print("Non-member data range:", np.min(nonmember), np.max(nonmember))

    alpha = 0.7
    bins = 30
    color1 = 'salmon'
    color2 = 'royalblue'
    edgecolor = 'black'

    plt.hist(member, bins=bins, alpha=alpha, label='member', color=color1, edgecolor=edgecolor)
    plt.hist(nonmember, bins=bins, alpha=alpha, label='non-member', color=color2, edgecolor=edgecolor)

    plt.xlabel('LOSS')
    plt.legend()
    # plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # plt.xlim(-0.05, 1.05)

    plt.xticks()
    plt.yticks()
    plt.tight_layout()
    # plt.gca().set_yticks([])
    plt.show()


def metric_attack_evaluation(args, targetmodel, shadowmodel, target_data, shadow_data, num_classes, mu=None):
    targetmodel.eval()
    PLoss = []
    Maximum = []
    ModifyEntropy = []
    Gap = []

    mem_groundtruth = np.ones(int(len(target_data) / 2))
    non_groundtruth = np.zeros(int(len(target_data) / 2))
    groundtruth = np.concatenate((mem_groundtruth, non_groundtruth))
    with torch.no_grad():
        for data in target_data:
            data = data.cuda()
            target = data.y.cpu()
            output = targetmodel(data.x, data.edge_index, data.batch)
            posterior = F.softmax(output, dim=1).cpu()  # 获取阴影模型后验概率
            pred = np.argmax(posterior)

            ce_criterion = nn.CrossEntropyLoss()

            # celoss = ce_criterion(posterior,target)
            celoss = ce_criterion(posterior, torch.LongTensor([pred]))

            PLoss.append(-1 * celoss.item())

            if pred.item() != target:
                Gap.append(0)
            else:
                Gap.append(1)

            # baseline1: max-confidence
            Maximum.append(torch.max(posterior))

            # baseline:modifyEntropy
            encoding_y = F.one_hot(target, num_classes=num_classes)
            ModifyEntropy.extend(
                -1 * _Mentr(np.asarray(posterior), np.asarray(encoding_y)))

    predictions_Loss = np.asarray(PLoss)
    predictions_Maximum = np.asarray(Maximum)
    predictions_ModifyEntropy = np.asarray(ModifyEntropy)
    predictions_GAP = np.asarray(Gap)

    print('===>Gap Attack')
    GAP_metrics = cal_metric_threshold(groundtruth, predictions_GAP, 1e-8, metric='gap')
    # print(classification_report(groundtruth, predictions_GAP))

    print('===>Maximum')
    _, tMaximum = estimate_threshold_with_shadow(shadowmodel, shadow_data, num_classes, metric='Maximum')
    Maximum_metrics = cal_metric_threshold(groundtruth, predictions_Maximum, tMaximum, metric='Maximum')

    print('===> CrossEntropy Loss')
    _, tCE = estimate_threshold_with_shadow(shadowmodel, shadow_data, num_classes, metric='CELoss')
    Loss_metrics = cal_metric_threshold(groundtruth, predictions_Loss, tCE, metric='CELoss')

    print('===>ModifyEntropy')
    _, tMEs = estimate_threshold_with_shadow(shadowmodel, shadow_data, num_classes, metric='ModifyEntropy')
    ModifyEntropy_metrics = cal_metric_threshold_mentr(groundtruth, predictions_ModifyEntropy, target_data, tMEs)

    print('-' * 50)

    score_path = args.result_path + 'score.txt'
    title = ['Loss', 'Entropy', 'Maximum', 'ModifyEntropy', 'GAP']
    now = datetime.now()
    timestamp = now.strftime('%H%M%S')

    for i, data in enumerate(
            [predictions_Loss, predictions_Maximum, predictions_ModifyEntropy, Gap]):
        path = args.root_path + f'score/{title[i]}/'
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path + '{}.npy'.format(timestamp), data)


    with open(score_path, 'a') as f:
        f.write("GAP:\n{}\n".format(Gap))
        f.write("prediction Loss:\n{}\n".format(predictions_Loss))
        f.write("prediction Maximum:\n{}\n".format(predictions_Maximum))
        f.write("prediction ModifyEntropy:\n{}\n".format(predictions_ModifyEntropy))

        f.write('\n')
        f.write('=' * 80 + '\n')

    f.close()
    if args.weight_decay == 0:
        result_path = args.result_path + f'result.txt'
        if args.perturb_type == 'fg':
            result_path = args.result_path + f'result_fg.txt'
    else:
        result_path = args.result_path + f'result_wd={args.weight_decay}.txt'
    with open(result_path, 'a') as f:
        f.write(
            "GAP:\nauc:{},acc:{}\tprecision:{}\trecall:{}\tf1:{}\tfpr@0.1tpr:{}\tprecision-marco:{}\trecall-macro:{}\tf1-marco:{}\n".format(
                GAP_metrics[0], GAP_metrics[1],
                GAP_metrics[2],
                GAP_metrics[3],
                GAP_metrics[4], GAP_metrics[5], GAP_metrics[6], GAP_metrics[7], GAP_metrics[8]))

        f.write(
            "prediction Loss:\nauc:{},acc:{}\tprecision:{}\trecall:{}\tf1:{}\tfpr@0.1tpr:{}\tprecision-marco:{}\trecall-macro:{}\tf1-marco:{}\n".format(
                Loss_metrics[0], Loss_metrics[1],
                Loss_metrics[2],
                Loss_metrics[3],
                Loss_metrics[4], Loss_metrics[5], Loss_metrics[6], Loss_metrics[7], Loss_metrics[8]))
        f.write(
            "prediction Maximum:\nauc:{},acc:{}\tprecision:{}\trecall:{}\tf1:{}\tfpr@0.1tpr:{}\tprecision-marco:{}\trecall-macro:{}\tf1-marco:{}\n".format(
                Maximum_metrics[0], Maximum_metrics[1],
                Maximum_metrics[2],
                Maximum_metrics[3],
                Maximum_metrics[4], Maximum_metrics[5], Maximum_metrics[6], Maximum_metrics[7], Maximum_metrics[8]))

        f.write(
            "prediction ModifyEntropy:\nauc:{},acc:{}\tprecision:{}\trecall:{}\tf1:{}\tfpr@0.1tpr:{}\tprecision-marco:{}\trecall-macro:{}\tf1-marco:{}\n".format(
                ModifyEntropy_metrics[0], ModifyEntropy_metrics[1],
                ModifyEntropy_metrics[2],
                ModifyEntropy_metrics[3],
                ModifyEntropy_metrics[4], ModifyEntropy_metrics[5], ModifyEntropy_metrics[6],
                ModifyEntropy_metrics[7], ModifyEntropy_metrics[8]))
        f.write('\n')

    f.close()

    return Loss_metrics, Maximum_metrics, ModifyEntropy_metrics, GAP_metrics
