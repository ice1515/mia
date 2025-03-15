import numpy as np
import torch
from torch_geometric.loader import DataLoader

from utils import plot_distribution, determine_threshold, \
    cal_metric_threshold_target


def stastic_score(args, scores, labels):
    scores_label_m = [data for data, y in zip(scores, labels) if y == 1]
    scores_label_nm = [data for data, y in zip(scores, labels) if y == 0]
    in_score = scores_label_m
    print('member max:{:6f},min:{:6f},average:{:6f},medium:{:.6f}'.format(np.max(in_score), np.min(in_score),
                                                                          np.average(in_score), np.median(in_score)))
    out_score = scores_label_nm
    print('nonmember max:{:6f},min:{:6f},average:{:6f},medium:{:.6f}'.format(np.max(out_score), np.min(out_score),
                                                                             np.average(out_score),
                                                                             np.median(out_score)))
    print()

    return np.average(in_score), np.min(in_score)


def getScores(args, model, data, scaler, device, groundtruth):
    scores = []
    for idx, data_i in enumerate(data):
        score_data = noise_perturbation(args, model, data_i, device, scaler)
        scores.append(score_data)
        print(
            'generate noisy graph:{}/{}\tscore:{}\tnodes:{}\tedges:{}\tg-label:{}\tn/m:{}'.format(
                idx + 1,
                len(data),
                score_data,
                data_i.num_nodes,
                data_i.num_edges,
                data_i.y.item(),
                groundtruth[idx]))
    return np.asarray(scores)


def estimate_threshold_with_shadow_p(args, shadowmodel, shadow_data, groundtruth, device, scaler):
    score_s = getScores(args, shadowmodel, shadow_data, scaler, device, groundtruth)
    estimate_t, estimate_acc, auc, tpr_fpr = determine_threshold(args, groundtruth, score_s, scaler)
    indicator_avg, indicator_min = stastic_score(args, score_s, groundtruth)
    return estimate_t, estimate_acc, indicator_avg, indicator_min, auc


def calculate_entropy(predictions, classes):
    predictions_total = [0] * classes
    for i in predictions:
        predictions_total[i] += 1
    counts = torch.tensor(predictions_total, dtype=torch.float32)
    posterior = counts / len(predictions)
    entropy = -torch.sum(posterior * torch.log(posterior + 1e-10))
    return entropy


def noise_all(data, scaler):
    node_num, feat_dim = data.x.size()
    noise = scaler * np.random.uniform(0.1, 0.5, size=(node_num, feat_dim))
    sign_mask = np.random.randint(2, size=(node_num, feat_dim)) * 2 - 1
    noise *= sign_mask
    data.x += torch.tensor(noise, dtype=data.x.dtype)
    return data


def nonzero_noise_valid(data, nonzero_mask, scaler, noise_range=(0.1, 0.5)):
    node_num, feat_dim = data.x.size()
    noise = torch.zeros((node_num, feat_dim), dtype=data.x.dtype)
    scaler_range = (scaler * noise_range[0], scaler * noise_range[1])
    noise[nonzero_mask] = torch.empty(nonzero_mask.sum(), dtype=data.x.dtype).uniform_(*scaler_range)

    sign_mask = torch.zeros((node_num, feat_dim), dtype=data.x.dtype)
    sign_mask[nonzero_mask] = torch.randint(0, 2, size=(nonzero_mask.sum(),), dtype=data.x.dtype) * 2 - 1

    noise *= sign_mask
    data.x[nonzero_mask] += noise[nonzero_mask]
    return data


def noise_perturbation(args, model, data, device, scaler):
    model.eval()
    data = data.to(device)
    label = model(data.x, data.edge_index, data.batch).max(1)[1].cpu().detach().numpy()
    ground_label = data.y.item()
    if ground_label == label:
        data_perturb = []
        if args.perturb_type == 'fr':
            data = data.cpu()
            for i in range(args.noise_number):
                newdata = data.clone()
                newdata = noise_all(newdata, scaler)
                data_perturb.append(newdata)
        elif args.perturb_type == 'fnz':
            data = data.cpu()
            nonzero_mask = data.x != 0
            for i in range(args.noise_number):
                newdata = data.clone()
                newdata = nonzero_noise_valid(newdata, nonzero_mask, scaler)
                data_perturb.append(newdata)
        preds = []
        dataloader = DataLoader(data_perturb, batch_size=100, shuffle=False)
        with torch.no_grad():
            for data in dataloader:
                data = data.to(device)
                pred = model(data.x, data.edge_index, data.batch).max(1)[1].cpu().detach().numpy()
                preds.extend(pred)

        if args.metric == 'acc':
            labels = np.repeat(label, len(data_perturb))
            score = np.mean(preds == labels)
        elif args.metric == 'entropy':
            score = calculate_entropy(preds, args.num_classes)

    else:
        if args.metric == 'acc':
            score = 0
        elif args.metric == 'entropy':
            score = 100
    return score


# 估计扰动率
def estimate_ratio(args, shadowmodel, shadowdata, groundtruth, device):
    print('[*] estimate hyperparameters with shadow...')
    dataset_params = {
        'DD': {
            'fnz': {'scaler': 0.1, 'step': 0.1},
            'fr': {'scaler': 0.01, 'step': 0.01},
        },
        'ENZYMES': {
            'fnz': {'scaler': 1, 'step': 0.1},
            'fr': {'scaler': 1, 'step': 0.1},
        },
        'PROTEINS_full': {
            'fnz': {'scaler': 1, 'step': 0.1},
            'fr': {'scaler': 1, 'step': 0.1},
        },
    }
    params = dataset_params[args.dataset]
    config = params[args.perturb_type]
    scaler = config.get('scaler')
    step = config.get('step')

    if args.scaler != None:
        scaler = args.scaler
        step = scaler + 2.0

    scaler_lst = []
    auc_lst = []
    s_acc_lst = []
    s_score_lst = []

    s_t_lst = []

    new_scaler = scaler
    max_auc = 0
    max_acc = 0
    while True:
        new_scaler = round(new_scaler, 2)
        print(f'======================scaler:{new_scaler} step:{step}===============================')
        t, acc, indicator_avg, indicator_min, auc = estimate_threshold_with_shadow_p(
            args, shadowmodel, shadowdata, groundtruth, device, new_scaler
        )
        if auc < max_auc and acc < max_acc:
            break
        else:
            max_auc = auc
            max_acc = acc

            s_acc_lst.append(acc)
            s_t_lst.append(t)
            scaler_lst.append(new_scaler)
            auc_lst.append(auc)

            new_scaler += step

    s_acc_lst = np.array(s_acc_lst)
    print(scaler_lst, auc_lst, s_acc_lst, s_t_lst, sep='\n')

    max_sco_idx = len(s_score_lst) - 1
    max_std = scaler_lst[max_sco_idx]
    max_t = s_t_lst[max_sco_idx]
    max_acc = s_acc_lst[max_sco_idx]
    return max_std, max_t, max_acc


def perturbation_attack(args, targetmodel, targetdata, shadowdata, shadowmodel, device):
    print('[*] Perturbation type : {}'.format(args.perturb_type))
    print('[*] Calculate type : {}'.format(args.metric))

    mem_groundtruth = np.ones(int(len(targetdata) / 2))
    non_groundtruth = np.zeros(int(len(targetdata) / 2))
    groundtruth = np.concatenate((mem_groundtruth, non_groundtruth))

    t = -1

    # ==================================step1:Estimate Hyperparameters=============================================
    if args.estimate_ratio:
        scaler, t, acc = estimate_ratio(args, shadowmodel, shadowdata, groundtruth, device)
        args.scaler = scaler
    else:
        scaler = args.scaler
    print('[*] Perturbation type : {}'.format(args.perturb_type))

    print(f'[*] scaler:{scaler}\tloc:{args.perturb_type}\t' + '=' * 15)
    # ================================= step 2:Determine theshold if not estimate ratio===================================================================
    if t == -1:
        t, _, _, _, _ = estimate_threshold_with_shadow_p(args, shadowmodel, shadowdata, groundtruth, device, scaler)
    print(f'[*]estimate threshold:{t},scaler:{scaler}')

    # ==================================step 3: Attack target model====================================================================
    scores = getScores(args, targetmodel, targetdata, scaler, device, groundtruth)
    if not args.estimate_ratio:
        np.save(args.result_path + '{}.npy'.format(args.scaler), scores)

    scaler_lst = [scaler]
    t_lst = [t]
    _, AUC, accuracy, precision, recall, fscore, low_fpr_tpr, precision_macro, recall_marco, f1score_macro = cal_metric_threshold_target(
        args, groundtruth, scores, t)
    dict_results = [AUC, accuracy, precision, recall, fscore, low_fpr_tpr, precision_macro, recall_marco,
                    f1score_macro]
    stastic_score(args, scores, groundtruth)

    # ====================Data Processing 1: Plot the distribution of membership scores================================================
    half = len(scores) // 2
    if args.draw_plot:
        plot_distribution(args.t_pic_path, scores[:half], scores[half:],
                          'scaler={}_{}_{}'.format(scaler, args.split_number, args.metric))

    # ==============================Data Processing 2: Save the results of each perturbation====================================================
    if args.write:
        with open(args.result_path + f'result.txt', 'a') as f:
            f.write(
                'seed:{}\tsplit:{}\t perturb_type:{} metric:{}, threshold-select:{},noise number:{}\n '.format(
                    args.seed,
                    args.split_number,
                    args.perturb_type,
                    args.metric,
                    args.perturb_type,
                    args.noise_number))
            if args.perturb_type in ['fr', 'fnz']:
                f.write(
                    '[!] scaler={},t={},auc:{},acc:{},precision:{},recall:{},f1:{},TPR@FPR=0.01:{},precision_macro:{}, recall_marco:{} , fscore_macro:{} \n\n'.format(
                        scaler_lst, t_lst, AUC, accuracy, precision, recall, fscore,
                        low_fpr_tpr, precision_macro, recall_marco, f1score_macro))
            f.write('=' * 80 + '\n')

        stastic_score(args, scores, groundtruth)

    # ====================Data Processing 3: Print the results of each perturbation====================================================

    if args.perturb_type in ['fr', 'fnz']:
        print(
            '[!] scaler={},t={},auc:{},acc:{},precision:{},recall:{},f1:{},TPR@FPR=0.01:{},precision_macro:{}, recall_marco:{} , fscore_macro:{} \n\n'.format(
                scaler_lst, t_lst, AUC, accuracy, precision, recall, fscore,
                low_fpr_tpr, precision_macro, recall_marco, f1score_macro))

    return dict_results
