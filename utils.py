import statistics as st

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, \
    classification_report, auc, precision_recall_fscore_support
from sklearn.cluster import KMeans



def determine_threshold(args, S_label_list, S_score_lst, stddev):
    max_acc, max_acc_pre, max_acc_rec, max_acc_f1, tpr_fpr = 0, 0, 0, 0, 0
    max_acc_threshold = 0

    y_true = np.array(S_label_list)
    y_score = np.asarray(S_score_lst)
    if args.metric == 'acc':
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        AUC = auc(fpr, tpr)
    elif args.metric == 'entropy':
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=0)
        AUC = auc(fpr, tpr)
    for thres in thresholds:
        if args.metric == 'acc':
            y_pred = [1 if m > thres else 0 for m in y_score]
        elif args.metric == 'entropy':
            y_pred = [1 if m < thres else 0 for m in y_score]

        ev_acc = accuracy_score(y_true, y_pred)

        if ev_acc > max_acc:
            max_acc = ev_acc
            max_acc_pre = precision_score(y_true, y_pred, zero_division=0)
            max_acc_rec = recall_score(y_true, y_pred, zero_division=0)
            max_acc_f1 = f1_score(y_true, y_pred, zero_division=0)
            max_acc_threshold = thres

    print("scaler={},selection t:{},auc:{},acc:{}, pre:{}, re:{}, f1:{},tpr-fpr:{:.6f}\n".format(
        stddev, max_acc_threshold, AUC, max_acc, max_acc_pre, max_acc_rec, max_acc_f1, tpr_fpr))

    with open(args.result_path + f'result.txt', 'a') as f:
        f.write("scaler={},selection t:{},auc:{},acc:{}, pre:{}, re:{}, f1:{},tpr-fpr:{:.6f}\n".format(
            stddev, max_acc_threshold, AUC, max_acc, max_acc_pre, max_acc_rec, max_acc_f1, tpr_fpr))
    f.close()
    return max_acc_threshold, max_acc, AUC, tpr_fpr






# 评估目标模型成员与非成员
def cal_metric_threshold_target(args, y_true, y_score, t):
    from bisect import bisect_left
    y_true = np.array(y_true)
    if args.metric == 'acc':
        y_pred = [1 if m > t else 0 for m in y_score]
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    elif args.metric == 'entropy':
        y_pred = [1 if m < t else 0 for m in y_score]
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=0)

    AUC = auc(fpr, tpr)
    # plot_auc(roc,fpr, tpr, 'perturb')
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    fscore = f1_score(y_true, y_pred, zero_division=0)
    print(classification_report(y_true, y_pred))

    fp_list = [0.1]
    tp_list = []
    for item in fp_list:
        fp_index = bisect_left(fpr, item)
        if fp_index == 0:
            tp_list.append(tpr[0])
        elif fp_list == len(fpr):
            raise ValueError("Give fpr larger than 1.0")
        else:
            tp_list.append(tpr[fp_index - 1])
    low_fpr_tpr = tp_list[0]

    precision_macro, recall_marco, fscore_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    return y_pred, AUC, accuracy, precision, recall, fscore, low_fpr_tpr, precision_macro, recall_marco, fscore_macro


def cal_metric_threshold_label(args, label_lst, y_true, y_score, t_lst):
    from bisect import bisect_left
    y_true = np.array(y_true)
    if args.metric == 'acc':
        y_pred = [1 if score > t_lst[label] else 0 for score, label in zip(y_score, label_lst)]
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    elif args.metric == 'entropy':
        y_pred = [1 if score < t_lst[label] else 0 for score, label in zip(y_score, label_lst)]
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=0)

    AUC = auc(fpr, tpr)
    # plot_auc(roc,fpr, tpr, 'perturb')
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    fscore = f1_score(y_true, y_pred, zero_division=0)
    print(classification_report(y_true, y_pred))

    fp_list = [0.1]
    tp_list = []
    for item in fp_list:
        fp_index = bisect_left(fpr, item)
        if fp_index == 0:
            tp_list.append(tpr[0])
        elif fp_list == len(fpr):
            raise ValueError("Give fpr larger than 1.0")
        else:
            tp_list.append(tpr[fp_index - 1])
    low_fpr_tpr = tp_list[0]

    precision_macro, recall_marco, fscore_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    return y_pred, AUC, accuracy, precision, recall, fscore, low_fpr_tpr, precision_macro, recall_marco, fscore_macro


def check_binary_features(data):
    binary_features = []
    non_binary_features = []

    for i, graph in enumerate(data):
        x = graph.x.numpy() if graph.x is not None else None

        if x is not None:
            for j in range(x.shape[1]):  # 遍历所有特征
                unique_values = np.unique(x[:, j])
                if set(unique_values).issubset({0, 1}):
                    binary_features.append(j)
                else:
                    non_binary_features.append(j)
    binary_features, non_binary_features = set(binary_features), set(non_binary_features)
    print(binary_features, non_binary_features, sep='\n')
    filtered_binary_features = binary_features - non_binary_features
    print('binary dim:', filtered_binary_features)

    return binary_features, non_binary_features




def plot_distribution(path, train_scores, test_scores, title, colorm='darkblue', colorn='red'):
    plt.figure(figsize=(10, 5))
    alpha = 0.5
    bins = 30
    plt.figure(figsize=(8, 6))
    train_scores = list(filter(lambda x: x != 0, train_scores))


    plt.hist(train_scores, bins=bins, alpha=alpha, label='Train Set', color=colorm, edgecolor='black')

    plt.hist(test_scores, bins=bins, alpha=alpha, label='Test Set', color=colorn, edgecolor='black')
    plt.xlabel('Confidence Score')
    plt.ylabel('Number')
    plt.title(title)
    plt.legend()

    plt.tight_layout()
    plt.savefig(path + '/' + title + '.png')

    # plt.show()





def dataToList(data1, data2):
    data = [a for a in data1]
    tmp = [a for a in data2]
    data.extend(tmp)
    return data


def process_result(path, result_lst, type, args):
    import csv

    '''
    metric-based method
    '''
    result_lst = np.array(result_lst)
    #  AUC,accuracy, precision, recall, fscore, low_fpr_tpr,precision_marco,recall_marco,fscore_macro
    avg_attack_auc = np.mean(result_lst[:, 0])
    avg_attack_acc = np.mean(result_lst[:, 1])
    avg_attack_precision = np.mean(result_lst[:, 2])
    avg_attack_recall = np.mean(result_lst[:, 3])
    avg_attack_f1 = np.mean(result_lst[:, 4])
    avg_low_fpr_tpr = np.mean(result_lst[:, 5])
    avg_macro_precision = np.mean(result_lst[:, 6])
    avg_macro_recall = np.mean(result_lst[:, 7])
    avg_macro_f1 = np.mean(result_lst[:, 8])

    attack_auc_stdev = np.std(result_lst[:, 0])
    attack_acc_stdev = np.std(result_lst[:, 1])
    attack_precision_stdev = np.std(result_lst[:, 2])
    attack_recall_stdev = np.std(result_lst[:, 3])
    attack_f1_stdev = np.std(result_lst[:, 4])
    attack_fpr_tpr_stdev = np.std(result_lst[:, 5])
    macro_precision_stdev = np.std(result_lst[:, 6])
    macro_recall_stdev = np.std(result_lst[:, 7])
    macro_f1_stdev = np.std(result_lst[:, 8])
    max_f1 = max(result_lst[:, 4])
    min_f1 = min(result_lst[:, 4])
    max_acc = max(result_lst[:, 1])
    min_acc = min(result_lst[:, 1])
    max_f1_macro = max(result_lst[:, 8])
    min_f1_macro = min(result_lst[:, 8])

    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([type])

        # 写入统计数据
        writer.writerow(
            ["auc", "acc", "precision", "recall", "f1", "fpr_tpr", "precision_macro", "recall_macro",
             "f1_macro", "max f1", "min_f1", "max acc", "min acc", "max f1-macro", "min f1-macro"])

        writer.writerow([
            avg_attack_auc,
            avg_attack_acc,
            avg_attack_precision,
            avg_attack_recall,
            avg_attack_f1,
            avg_low_fpr_tpr,
            avg_macro_precision, avg_macro_recall, avg_macro_f1,
            max_f1,
            min_f1, max_acc, min_acc, max_f1_macro, min_f1_macro
        ])

        if len(result_lst[:, 0]) > 1:
            writer.writerow([
                attack_auc_stdev,
                attack_acc_stdev,
                attack_precision_stdev,
                attack_recall_stdev,
                attack_f1_stdev,
                attack_fpr_tpr_stdev,
                macro_precision_stdev,
                macro_recall_stdev,
                macro_f1_stdev
            ])

        else:
            print('only attack  once')
        writer.writerow([])
    print("=" * 25 + f'{type}' + "=" * 25)
    print("Max f1:{},Min f1 :{}".format(np.max(avg_attack_f1), np.min(avg_attack_f1)))

    print(
        "Attack auc:{},acc:{},precision stdev:{}, recall stdev:{}, and f1 stdev:{}, Macro precision stdev:{}, recall stdev:{}, and f1 stdev:{}".format(
            attack_auc_stdev, attack_acc_stdev, attack_precision_stdev, attack_recall_stdev, attack_f1_stdev,
            macro_precision_stdev, macro_recall_stdev, macro_f1_stdev))

    print(
        "Average attack auc:{},acc:{},precision:{}, recall:{}, f1:{} and low_fpr_tpr:{}, Average marco precision:{}, recall:{} and f1:{}".format(
            avg_attack_auc, avg_attack_acc, avg_attack_precision, avg_attack_recall, avg_attack_f1, avg_low_fpr_tpr,
            avg_macro_precision, avg_macro_recall, avg_macro_f1))






def check_binary_features(args, dataset):
    from torch_geometric.datasets import TUDataset
    dataset = TUDataset(root='dataset', name=dataset, use_edge_attr=False,
                        use_node_attr=args.use_node_attr)

    binary_feature_dims = []
    features = []
    for data in dataset:
        node_features = data.x
        features.extend(node_features.tolist())
    features = np.array(features)
    for i in range(len(features[0])):
        feat_dim = np.unique(features[:, i])
        print(f'idx={i},feature:{feat_dim}')
        if len(feat_dim) == 2 and feat_dim[0] == 0 and feat_dim[1] == 1:
            binary_feature_dims.append(i)
    args.binary_dim = binary_feature_dims





