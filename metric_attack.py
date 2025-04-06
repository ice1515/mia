import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc
from sklearn.metrics import roc_curve, \
    classification_report
from torch import nn

from MotifyEntropy import _Mentr
from estimate_threshold import estimate_threshold_with_shadow, cal_metric_threshold

''' code implemented from
    Reference1:"Membership Leakage in Label-Only Exposures" ""
    github1:https://github.com/zhenglisec/Label-Only-MIA/blob/main/attack.py
    
    Reference2:"Adapting Membership Inference Attacks to GNN for Graph Classification: Approaches and Implications"
    github2:https://github.com/TrustworthyGNN/MIA-GNN
'''


def metric_attack_evaluation(args, targetmodel, shadowmodel, target_data, shadow_data, num_classes):
    Loss = []
    Entropy = []
    Maximum = []
    ModifyEntropy = []
    Gap = []
    mem_groundtruth = np.ones(int(len(target_data) / 2))
    non_groundtruth = np.zeros(int(len(target_data) / 2))
    groundtruth = np.concatenate((mem_groundtruth, non_groundtruth))
    with torch.no_grad():
        for data in target_data:
            data, target = data.cuda(), data.y.cuda()

            output = targetmodel(data.x, data.edge_index, data.batch)
            posterior = F.softmax(output, dim=1).cpu()  # 获取阴影模型后验概率
            # label = posterior.max(1)[1]
            label = np.argmax(posterior)

            ce_criterion = nn.CrossEntropyLoss()

            celoss = ce_criterion(posterior, torch.LongTensor([label]))
            Loss.append(celoss.item())

            if label.item() != target.item():
                Gap.append(0)
            else:
                Gap.append(1)

            # baseline1: max-confidence
            Maximum.append(torch.max(posterior).item())

            # baseline2:normalized entropy
            entropy = -1 * torch.sum(torch.mul(posterior, torch.log(posterior)))
            if str(entropy.item()) == 'nan':  # 预处理
                Entropy.append(1e-100)
            else:
                Entropy.append(entropy.item())

            # baseline:modifyEntropy
            encoding_y = F.one_hot(target, num_classes=num_classes)
            ModifyEntropy.extend(_Mentr(np.asarray(posterior.detach().cpu()), np.asarray(encoding_y.detach().cpu())))

    print('GAP:', Gap)

    predictions_Loss = np.asarray(Loss)
    predictions_Entropy = np.asarray(Entropy)
    predictions_Maximum = np.asarray(Maximum)
    predictions_ModifyEntropy = np.asarray(ModifyEntropy)
    predictions_GAP = np.asarray(Gap)



    fpr0, tpr0, _ = roc_curve(groundtruth, predictions_Loss, pos_label=0, drop_intermediate=False)
    AUC_Loss = auc(fpr0, tpr0)

    fpr, tpr, _ = roc_curve(groundtruth, predictions_Entropy, pos_label=0, drop_intermediate=False)
    AUC_Entropy = auc(fpr, tpr)

    fpr, tpr, _ = roc_curve(groundtruth, predictions_Maximum, pos_label=1, drop_intermediate=False)
    AUC_Maximum = auc(fpr, tpr)

    fpr, tpr, _ = roc_curve(groundtruth, predictions_ModifyEntropy, pos_label=0, drop_intermediate=False)
    AUC_ModifyEntropy = auc(fpr, tpr)

    fpr, tpr, _ = roc_curve(groundtruth, predictions_GAP, pos_label=1, drop_intermediate=False)
    AUC_GAP = auc(fpr, tpr)

    print('-' * 50)
    print('AUC_Loss:{:.4f}, AUC_Entropy:{:4f},AUC_Maximum:{:.4f},AUC_ModifyEntropy:{:.4f}'.format(AUC_Loss, AUC_Entropy,
                                                                                                  AUC_Maximum,
                                                                                                  AUC_ModifyEntropy))


    if args.estimate_threshold:
        print('===>Gap Attack')
        GAP_metrics = cal_metric_threshold(groundtruth, predictions_GAP, 0, metric='gap')
        print(classification_report(groundtruth, predictions_GAP))

        print('===>CrossEntropy Loss')
        _, tCE = estimate_threshold_with_shadow(shadowmodel, shadow_data, num_classes, metric='CELoss')
        Loss_metrics = cal_metric_threshold(groundtruth, predictions_Loss, tCE, metric='CELoss')

        print('===>Maximum')
        _, tMaximum = estimate_threshold_with_shadow(shadowmodel, shadow_data, num_classes, metric='Maximum')
        Maximum_metrics = cal_metric_threshold(groundtruth, predictions_Maximum, tMaximum, metric='Maximum')

        print('===>NormalizedEntropy')
        _, tNE = estimate_threshold_with_shadow(shadowmodel, shadow_data, num_classes, metric='NormalizedEntropy')
        Entropy_metrics = cal_metric_threshold(groundtruth, predictions_Entropy, tNE, metric='NormalizedEntropy')

        print('===>ModifyEntropy Loss')
        _, tME = estimate_threshold_with_shadow(shadowmodel, shadow_data, num_classes, metric='ModifyEntropy')
        ModifyEntropy_metrics = cal_metric_threshold(groundtruth, predictions_ModifyEntropy, tME,
                                                     metric='ModifyEntropy')

        print('-' * 50)



    else:
        Loss_metrics = [AUC_Loss, 0, 0, 0, 0, 0]
        Maximum_metrics = [AUC_Maximum, 0, 0, 0, 0, 0]
        Entropy_metrics = [AUC_Entropy, 0, 0, 0, 0, 0]
        ModifyEntropy_metrics = [AUC_ModifyEntropy, 0, 0, 0, 0, 0]
        GAP_metrics = [AUC_GAP, 0, 0, 0, 0]

    write = True

    if write:
        score_path = args.result_path + 'score.txt'
        title = ['Loss', 'Entropy', 'Maximum', 'ModifyEntropy', 'GAP']
        now = datetime.now()
        timestamp = now.strftime('%H%M%S')

        for i, data in enumerate(
                [predictions_Loss, predictions_Entropy, predictions_Maximum, predictions_ModifyEntropy, Gap]):
            path = args.root_path + f'score/{title[i]}/'
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(path + '{}.npy'.format(timestamp), data)

        result_path = args.result_path + f'result.txt'
        with open(score_path, 'a') as f:
            f.write("GAP:\n{}\n".format(Gap))
            f.write("prediction Loss:\n{}\n".format(predictions_Loss))
            f.write("prediction Entropy:\n{}\n".format(predictions_Entropy))
            f.write("prediction Maximum:\n{}\n".format(predictions_Maximum))
            f.write("prediction ModifyEntropy:\n{}\n".format(predictions_ModifyEntropy))

            f.write('\n')
            f.write('=' * 80 + '\n')

        f.close()

        with open(result_path, 'a') as f:
            f.write(
                "GAP:\nauc:{},acc:{}\tprecision:{}\trecall:{}\tf1:{}\tfpr@0.1tpr:{}\tprecision-marco:{}\trecall-macro:{}\tf1-marco:{}\n".format(
                    GAP_metrics[0], GAP_metrics[1],
                    GAP_metrics[2],
                    GAP_metrics[3],
                    GAP_metrics[4], GAP_metrics[5], GAP_metrics[6], GAP_metrics[7], GAP_metrics[8]))
            f.write(
                "prediction Entropy:\nauc:{},acc:{}\tprecision:{}\trecall:{}\tf1:{}\tfpr@0.1tpr:{}\tprecision-marco:{}\trecall-macro:{}\tf1-marco:{}\n".format(
                    Entropy_metrics[0], Entropy_metrics[1],
                    Entropy_metrics[2],
                    Entropy_metrics[3],
                    Entropy_metrics[4], Entropy_metrics[5], Entropy_metrics[6], Entropy_metrics[7], Entropy_metrics[8]))

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

    return Loss_metrics, Entropy_metrics, Maximum_metrics, ModifyEntropy_metrics, GAP_metrics
