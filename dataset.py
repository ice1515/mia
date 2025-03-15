from random import shuffle

import torch

from torch_geometric.datasets import TUDataset
import numpy as np

from utils import check_binary_features


def per_class_num(dataset, num_classes):
    classes_num_lst = [0 for _ in range(num_classes)]
    for data in dataset:
        classes_num_lst[data.y.item()] += 1
    return classes_num_lst


def graph_property(args, dataset, num_classes):
    max_nodes, min_nodes = 0, dataset[0].num_nodes
    max_edges, min_edges = 0, dataset[0].num_edges
    nodes = []
    edges = []

    for graph in dataset:
        if graph.num_nodes > max_nodes:
            max_nodes = graph.num_nodes
        elif graph.num_nodes < min_nodes:
            min_nodes = graph.num_nodes
        if graph.num_edges > max_edges:
            max_edges = graph.num_edges
        elif graph.num_edges < min_edges:
            min_edges = graph.num_edges
        nodes.append(graph.num_nodes)
        edges.append(graph.num_edges)
    print('graphs:', len(dataset))
    per_class_num(dataset, num_classes)
    print('min_nodes:', min_nodes, 'max_nodes:', max_nodes, 'average nodes:', sum(nodes) / len(dataset))
    print('min_edges:', min_edges, 'max_edges:', max_edges, 'average edges:', sum(edges) / len(dataset))
    args.avg_edges = sum(edges) / len(dataset)

def load_data_random(args):
    dataset_name = args.dataset
    dataset = TUDataset(root='dataset', name=dataset_name, use_edge_attr=False,
                        use_node_attr=args.use_node_attr)
    args.input_dim = dataset.num_features
    args.feat_dim = dataset.num_features
    args.output_dim = dataset.num_classes
    args.num_classes = dataset.num_classes

    print('Dataset:', per_class_num(dataset, dataset.num_classes))
    print("feature dim: ", args.input_dim)
    train_size, test_size = args.train_size, args.test_size

    index = list(range(len(dataset)))
    shuffle(index)
    n = int(len(dataset) * 0.5)
    train_index, test_index = index[:n], index[n:]
    print("Split Size:", len(train_index), len(test_index))

    target_train_index, shadow_train_index = train_index[:len(train_index) // 2], train_index[
                                                                                  len(train_index) // 2:]
    target_test_index, shadow_test_index = test_index[:len(test_index) // 2], test_index[
                                                                              len(test_index) // 2:]
    target_train_set, target_test_set, shadow_train_set, shadow_test_set = dataset[target_train_index], dataset[
        target_test_index], \
        dataset[shadow_train_index], dataset[shadow_test_index]
    selected_T_train_index = target_train_index[:train_size]
    selected_T_test_index = target_test_index[:test_size]

    selected_S_train_index = shadow_train_index[:train_size]
    selected_S_test_index = shadow_test_index[:test_size]

    selected_T_train_set = dataset[selected_T_train_index]
    selected_T_test_set = dataset[selected_T_test_index]
    selected_S_train_set = dataset[selected_S_train_index]
    selected_S_test_set = dataset[selected_S_test_index]

    print('Selected Training Size:{}, and Testing Size:{}'.format(len(selected_T_train_set), len(selected_T_test_set)))
    print('Selected Shadow Size:{}, Testing Size:{}'.format(len(selected_S_train_set),
                                                            len(selected_S_test_set)))

    print('target train size:{},test size:{}'.format(len(selected_T_train_set), len(selected_T_test_set)))
    print('shadow train size:{},test size:{}'.format(len(selected_S_train_set), len(selected_S_test_set)))

    print('Target train set:', per_class_num(selected_T_train_set, dataset.num_classes))
    print('Target test set:', per_class_num(selected_T_test_set, dataset.num_classes))
    print('Shadow train set:', per_class_num(selected_S_train_set, dataset.num_classes))
    print('Shadow test set:', per_class_num(selected_S_test_set, dataset.num_classes))
    print('Target train property:')
    graph_property(args, selected_T_train_set, dataset.num_classes)
    print('Shadow train property:')
    graph_property(args, selected_S_train_set, dataset.num_classes)

    data = {
        'T_train_set': selected_T_train_set,
        'T_test_set': selected_T_test_set,
        'S_train_set': selected_S_train_set,
        'S_test_set': selected_S_test_set,
    }
    torch.save(data, args.data_save_path + 'data.pt')
    print(f"data save: {args.data_save_path}")

    return selected_T_train_set, selected_T_test_set, selected_S_train_set, selected_S_test_set
