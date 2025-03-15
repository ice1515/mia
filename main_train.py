import argparse
import hashlib
import json
import os
import random

import numpy as np
import torch

from dataset import load_data_random
from train import get_model


def get_model_hash(model):
    model_state = model.state_dict()
    hash_obj = hashlib.sha256()
    for key in sorted(model_state.keys()):
        hash_obj.update(model_state[key].cpu().numpy().tobytes())
    return hash_obj.hexdigest()

def args_attack():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--result_path', type=str, help='result save path')
    parser.add_argument('--root_path', type=str, help='model save path')
    parser.add_argument('--data_save_path', type=str, help='dataset split  save path')
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
    parser.add_argument('--load', type=int, default=0, help='load pretrained model')

    # ================================training parameters======================================
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--init_lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--min_lr', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--epoch_interval', type=int)
    parser.add_argument('--step_size', type=int)
    parser.add_argument('--gamma', type=float)



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
    args.result_path = root_path
    data_save_path = f'dataset_split/{args.dataset}'
    args.data_save_path = data_save_path
    args.root_path = root_path
    path_lst = [root_path, data_save_path]
    for path in path_lst:
        if not os.path.exists(path):
            os.makedirs(path)
    print("[!] root path:", root_path)

    print('seed:{}'.format(args.seed))

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # =====================================load data ========================================
    target_train, target_test, shadow_train, shadow_test = load_data_random(args)

    # ======================================step2: train target model======================================================
    print('[!] model :{}'.format(args.model_name))

    get_model(args, target_train, target_test, 'target', device)
    get_model(args, shadow_train, shadow_test, 'shadow', device)


main()
