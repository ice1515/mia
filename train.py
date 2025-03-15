import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader

from model import GCN, GraphSAGE, GIN, GAT

'''
train/test  target and shadow model 
'''
from tqdm import tqdm
import time



def view_model_param(model):
    total_param = 0
    print("MODEL DETAILS:\n")
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    return total_param

def get_model(args, train_data, test_data, flag, device):

    model_path = args.root_path + args.model_name + '_' + str(args.epochs) + f'_{flag}.pkl'

    if args.model_name == 'GCN':
        model = GCN(args.num_hidden_layers, args.input_dim, args.hidden_dim, args.output_dim, args.readout,
                    args.dropout, args.batch_norm, args.residual)
    elif args.model_name == 'GIN':
        model = GIN(args.num_hidden_layers, args.input_dim, args.hidden_dim, args.output_dim, args.readout,
                    args.dropout, args.batch_norm, args.residual)
    elif args.model_name == 'GraphSage':
        model = GraphSAGE(args.num_hidden_layers, args.input_dim, args.hidden_dim, args.output_dim, args.readout,
                          args.dropout, args.batch_norm, args.residual)
    elif args.model_name == 'GAT':
        model = GAT(args.n_heads, args.num_hidden_layers, args.input_dim, args.hidden_dim, args.output_dim,
                    args.readout, args.dropout, args.batch_norm, args.residual)
    else:
        raise ValueError("[!] Unknown model type")
    print(f'{flag} model params',view_model_param(model))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    model = model.to(device)
    if not args.load:
        print(f"{flag} model:\n", model)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)

        train_loss = []
        with tqdm(range(args.epochs)) as t:
            for epoch in t:
                t.set_description('Epoch %d' % epoch)
                start = time.time()

                loss,train_acc = train_model(model, train_loader, optimizer, device)
                test_acc = test_model(model, test_loader, device)
                train_loss.append(loss)

                t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                               train_loss=loss, train_acc=train_acc,
                               test_acc=test_acc)
                if optimizer.param_groups[0]['lr'] < args.min_lr:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break
        torch.save(model.state_dict(), model_path)
        print(f'[*] Saved {flag} model,epoch={epoch + 1}')

        loss_path = args.root_path + args.model_name + '_' + str(args.epochs) + f'_{flag}.txt'
        with open(loss_path, 'w') as f:
            f.write(str(train_loss))
    else:
        # 装载训练好的模型，用于测试不同epoch下的攻击效果
        print(f'[*] Load trained {flag} model from {model_path}')
        model.load_state_dict(torch.load(model_path))

    train_acc = test_model(model, train_loader, device)
    test_acc = test_model(model, test_loader, device)

    print('[!] {} model train acc:{:0.4f},test acc:{:0.4f}, gap:{:0.4f} ,GAP:{}'.format(
        flag, train_acc, test_acc, train_acc - test_acc, 0.5 + (train_acc - test_acc) / 2))


    result_path = args.result_path + f'result.txt'
    with open(result_path, 'a') as f:
        f.write("{} {} train size:{},test size:{},train acc:{},test acc:{},gap:{},Gap attack acc:{}\n".format(flag,args.model_name,
                                                                                                           len(train_data),
                                                                                                           len(test_data),
                                                                                                           train_acc,
                                                                                                           test_acc,
                                                                                                           train_acc - test_acc,
                                                                                                           0.5 + (
                                                                                                                   train_acc - test_acc) / 2))
    f.close()

    return model, train_acc, test_acc


def train_model(model, loader, opt, device):
    model.train()
    loss_all = 0
    criterion = nn.CrossEntropyLoss()
    correct=0

    for iter, data in enumerate(loader):
        data = data.to(device)
        opt.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        pred = F.softmax(out, dim=1).argmax(dim=-1)
        loss = criterion(out, data.y.type(torch.long))
        loss.backward()
        loss_all += loss.detach().item()
        opt.step()

        correct += int((pred == data.y).sum())

    return loss_all / (iter + 1),correct / len(loader.dataset)


def test_model(model, loader, device):
    model.eval()
    correct = 0

    with torch.no_grad():
        for iter, data in enumerate(loader):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            pred = F.softmax(out, dim=1).argmax(dim=-1)
            correct += int((pred == data.y).sum())

    return correct / len(loader.dataset)
