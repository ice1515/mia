import argparse
import csv
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch_geometric.datasets import TUDataset

from metric_attack import metric_attack_evaluation
from perturbation import perturbation_attack
from train import get_model
from utils import process_result, dataToList


def view_model_param(model):
    total_param = 0
    print("MODEL DETAILS:\n")
    # print(model)
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    return total_param


def args_attack():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/TUs_graph_classification_GraphSage_DD.json')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--root_path', type=str, help='result and model save path')
    parser.add_argument('--result_path', type=str, help='attack result  save path')
    parser.add_argument('--t_pic_path', type=str, help='target data plot save path')
    parser.add_argument('--s_pic_path', type=str, help='shadow data plot  save path')
    parser.add_argument('--score_path', type=str, help='score save path')
    parser.add_argument('--dataset_split_path', type=str, help='dataset split  save path')

    parser.add_argument('--seed', type=int)

    # ================================dataset info======================================
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--split', type=bool)
    parser.add_argument('--feat_dim', type=bool)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--use_node_attr', type=bool)
    parser.add_argument('--test_size', type=int)
    parser.add_argument('--train_size', type=int)

    # ================================model parameters======================================
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--num_hidden_layers', type=int)
    parser.add_argument('--hidden_dim', type=int)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--residual', type=bool, default=False, help="Please give a value for residual")
    parser.add_argument('--dropout', type=float, default=0.0, help="Please give a value for dropout")
    parser.add_argument('--batch_norm', type=bool, default=False, help="Please give a value for batch_norm")
    parser.add_argument('--readout', type=str)
    parser.add_argument('--load', type=int, default=1, help='load pretrained model')

    # ================================training parameters======================================
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--init_lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--min_lr', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--epoch_interval', type=int)
    parser.add_argument('--step_size', type=int, default=2000)
    parser.add_argument('--gamma', type=float, default=0.98)

    # ================================attack parameters======================================
    parser.add_argument('--write', type=int, default=1, help='save statistic of attack results or not')
    parser.add_argument('--estimate_threshold', type=int, default=1, help='use shadow data estimate threshold or not')
    parser.add_argument('--estimate_ratio', type=int, default=1, help='use shadow data estimate perturb ratio or not')
    parser.add_argument('--draw_plot', type=int, default=1, help='whether draw score distribution or not')
    parser.add_argument('--attack_type', type=str, default='perturb')
    parser.add_argument('--scaler', type=float, help='random noise scaler')

    parser.add_argument('--perturb_type', type=str, default='fnz', help='perturbation location', )
    parser.add_argument('--noise_number', type=int, default=1000, help='perturb noise samples')
    parser.add_argument('--attack_num', type=int, default=5)
    parser.add_argument('--attack', type=int, default=1, help='execute membership inference attack or not')
    parser.add_argument('--metric', type=str, default="acc", choices=['acc', 'entropy'],
                        help='calculate confidence score')

    args = parser.parse_args()
    return args


def main():

    args = args_attack()
    device = torch.device('cuda:' + args.device if torch.cuda.is_available() else 'cpu')
    print(device)

    # ============================load parameters======================================

    with open(args.config) as f:
        config = json.load(f)

    if args.model_name is not None:
        config['model_name'] = args.model_name
    else:
        args.model_name = config['model_name']
    if args.dataset is not None:
        config['dataset'] = args.dataset
    else:
        args.dataset = config['dataset']
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    else:
        args.seed = params['seed']
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    else:
        args.epochs = params['epochs']
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    else:
        args.batch_size = params['batch_size']
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    else:
        args.init_lr = params['init_lr']
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    else:
        args.min_lr = params['min_lr']
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    else:
        args.weight_decay = params['weight_decay']
    if args.epoch_interval is not None:
        params['epoch_interval'] = int(args.epoch_interval)
    else:
        args.epoch_interval = params['epoch_interval']
    if args.train_size is not None:
        params['train_size'] = int(args.train_size)
    else:
        args.train_size = params['train_size']
    if args.test_size is not None:
        params['test_size'] = int(args.test_size)
    else:
        args.test_size = params['test_size']
    if args.num_hidden_layers is not None:
        params['num_hidden_layers'] = int(args.num_hidden_layers)
    else:
        args.num_hidden_layers = params['num_hidden_layers']
    if args.hidden_dim is not None:
        params['hidden_dim'] = int(args.hidden_dim)
    else:
        args.hidden_dim = params['hidden_dim']
    if args.residual is not None:
        params['residual'] = True if args.residual == 'True' else False
    else:
        args.residual = params['residual']
    if args.use_node_attr is not None:
        params['use_node_attr'] = True if args.use_node_attr == 'True' else False
    else:
        args.use_node_attr = params['use_node_attr']
    if args.readout is not None:
        params['readout'] = args.readout
    else:
        args.readout = params['readout']

    if args.dropout is not None:
        params['dropout'] = float(args.dropout)
    else:
        args.dropout = params['dropout']
    if args.batch_norm is not None:
        params['batch_norm'] = True if args.batch_norm == 'True' else False
    else:
        args.batch_norm = params['batch_norm']
    print(args)
    # ============================create dictionary======================================
    root_path = f'out/{args.dataset}_{args.model_name}/'

    s_pic_path = root_path + f'plot/s/{args.perturb_type}/'
    t_pic_path = root_path + f'plot/t/{args.perturb_type}/'
    data_path = root_path + 'score/'
    t_score_path = data_path + f'perturb/{args.perturb_type}/t/'
    s_score_path = data_path + f'perturb/{args.perturb_type}/s/'
    result_path = root_path + 'result/'
    data_save_path = f'dataset_split/{args.dataset}'

    args.t_score_path = t_score_path
    args.s_score_path = s_score_path
    args.root_path = root_path
    args.t_pic_path = t_pic_path
    args.s_pic_path = s_pic_path
    args.data_save_path = data_save_path
    args.result_path = result_path

    path_lst = [root_path, s_pic_path, t_pic_path, result_path, data_path, t_score_path, s_score_path]
    for path in path_lst:
        if not os.path.exists(path):
            os.makedirs(path)
    print("[!] root path:", root_path)

    print('seed:{}'.format(args.seed))

    # =====================================load data ========================================
    dataset = TUDataset(root='dataset', name=args.dataset, use_edge_attr=False,
                        use_node_attr=args.use_node_attr)

    args.input_dim = dataset.num_features
    args.feat_dim = dataset.num_features
    args.output_dim = dataset.num_classes
    args.num_classes = dataset.num_classes

    print('[!] input dim:', args.input_dim)
    data = torch.load(data_save_path + '/data.pt')
    target_train = data['T_train_set']
    target_test = data['T_test_set']
    shadow_train = data['S_train_set']
    shadow_test = data['S_test_set']
    targetdata = dataToList(target_train, target_test)
    shadowdata = dataToList(shadow_train, shadow_test)
    # ======================================step2: train target model======================================================
    print('[!] model :{}'.format(args.model_name))
    targetmodel, t_train_acc, t_test_acc = get_model(args, target_train, target_test, 'target',
                                                     device)
    shadowmodel, s_train_acc, s_test_acc = get_model(args, shadow_train, shadow_test, 'shadow',
                                                     device)

    # ================================ attack target model======================================

    Perturb = []
    GAP, Loss, Maximum, ModifyEntropy, f1_macro = [], [], [], [], []

    title = ['Perturb', 'GAP', 'ModifyEntropy', 'Loss', 'Maximum']
    path = args.result_path + f'results_{args.attack_num}.csv'
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([args])
        writer.writerow(
            ["auc", "acc", "precision", "recall", "f1", "fpr_tpr", "precision_macro", "recall_macro",
             "f1_macro", "attack type", "split number", "target model", "shadow model", 'noise_number', 'Gap_acc',
             'develop'])

        if args.attack:

            # ===========================baseline:  posterior-based attack=====================================
            LossResult, MaximumResult, ModifyEntropyResult, GapResult = metric_attack_evaluation(
                args, targetmodel, shadowmodel, targetdata, shadowdata,
                args.num_classes)
            Loss.append(LossResult)
            Maximum.append(MaximumResult)
            ModifyEntropy.append(ModifyEntropyResult)
            GAP.append(GapResult)
            for i in range(args.attack_num):
                args.split_number = i
                args.scaler = None
                print('=' * 25 + f'attack:{i + 1}/{args.attack_num}' + '=' * 25)
                start = datetime.now()
                result = perturbation_attack(args, targetmodel, targetdata, shadowdata, shadowmodel, device)

                Perturb.append(result)

                end = datetime.now()
                delta = end - start
                hours = delta.days * 24 + delta.seconds // 3600
                minutes = (delta.seconds % 3600) // 60
                seconds = (delta.seconds % 3600) % 60
                print(f"用时: {hours} 小时 {minutes} 分钟 {seconds} 秒")

                writer.writerow([
                    *result,
                    title[0],
                    i,
                    f' target:{t_train_acc}\t{t_test_acc:.5f}\t{t_train_acc - t_test_acc}',
                    f' shadow:{s_train_acc}\t{s_test_acc:.5f}\t{s_train_acc - s_test_acc}',
                    args.noise_number,
                    GapResult[1],
                    f'{(result[1] - GapResult[1]) * 100}%'
                ])
                writer.writerow([])

            print(' target:{}\t{}   shadow:{}\t{}'.format(t_train_acc, t_test_acc, s_train_acc,
                                                          s_test_acc))
    # ==================================== step4: process and save results=============================
    print("[!] Attack result:\n", Perturb)

    if args.attack_num > 1 and args.attack:
        for i, data in enumerate([Perturb, GAP, ModifyEntropy, Loss, Maximum]):
            np.save(args.result_path + '{}.npy'.format(title[i]), data)
            process_result(path, data, title[i], args)
            print()


main()
