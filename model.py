import torch
import torch.nn as nn

import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv, GINConv, SAGEConv, GATConv, GCN2Conv
from torch_geometric.nn.pool import global_mean_pool, global_max_pool, global_add_pool
import numpy as np


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y


class GCN(torch.nn.Module):
    def __init__(self, num_hidden_layers, num_node_features, hidden_channels, num_classes, readout, dropout,
                 batch_norm, residual=False):
        super(GCN, self).__init__()
        self.readout = readout
        self.batch_norm = batch_norm
        self.dropout = dropout

        if num_node_features != num_classes:
            self.residual = False
        if self.batch_norm:
            self.batchnorm_x = nn.BatchNorm1d(hidden_channels)
        if self.dropout:
            self.dropout = nn.Dropout(dropout)

        self.residual = residual
        self.input_layer = GCNConv(num_node_features, hidden_channels)

        self.hidden_layers = nn.ModuleList(
            [GCNConv(hidden_channels, hidden_channels) for _ in range(num_hidden_layers)])

        self.MLP_layer = MLPReadout(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.input_layer(x, edge_index)
        x = x.relu()

        for conv in self.hidden_layers:
            x_in = x
            x = conv(x, edge_index)
            if self.batch_norm:
                x = self.batchnorm_x(x)
            x = x.relu()
            if self.residual:
                x = x_in + x
            if self.dropout:
                x = self.dropout(x)
        if self.readout == 'max':
            x = global_max_pool(x, batch)
        elif self.readout == 'sum':
            x = global_add_pool(x, batch)
        elif self.readout == 'mean':
            x = global_mean_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)

        return self.MLP_layer(x)

    def predict(self, data, device):
        self.eval()
        data = data.to(device)
        out = self(data.x, data.edge_index, data.batch)
        score = F.softmax(out, dim=1)
        pred = score.argmax(dim=1)
        return pred


class GIN(torch.nn.Module):
    def __init__(self, num_hidden_layers, num_node_features, hidden_channels, num_classes, readout, dropout,
                 batch_norm, residual=False):
        super(GIN, self).__init__()
        self.readout = readout
        self.batch_norm = batch_norm
        self.dropout = dropout

        if num_node_features != num_classes:
            self.residual = False
        if self.batch_norm:
            self.batchnorm_x = nn.BatchNorm1d(hidden_channels)
        if self.dropout:
            self.dropout = nn.Dropout(dropout)

        self.residual = residual
        self.input_layer = GINConv(
            torch.nn.Sequential(torch.nn.Linear(num_node_features, hidden_channels), torch.nn.ReLU(),
                                torch.nn.Linear(hidden_channels, hidden_channels)))

        self.hidden_layers = nn.ModuleList(
            [GINConv(torch.nn.Sequential(torch.nn.Linear(hidden_channels, hidden_channels),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(hidden_channels, hidden_channels))) for
             _ in range(num_hidden_layers)])

        self.MLP_layer = MLPReadout(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.input_layer(x, edge_index)
        x = x.relu()

        for gin_layer in self.hidden_layers:
            x_in = x
            x = gin_layer(x, edge_index)
            if self.batch_norm:
                x = self.batchnorm_x(x)
            x = x.relu()
            if self.residual:
                x = x_in + x
            if self.dropout:
                x = self.dropout(x)
        if self.readout == 'max':
            x = global_max_pool(x, batch)
        elif self.readout == 'sum':
            x = global_add_pool(x, batch)
        elif self.readout == 'mean':
            x = global_mean_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)
        return self.MLP_layer(x)

    def predict(self, data, device):
        self.eval()
        data = data.to(device)
        out = self(data.x, data.edge_index, data.batch)
        score = F.softmax(out, dim=1)
        pred = score.argmax(dim=1)
        return pred


class GraphSAGE(nn.Module):
    def __init__(self, num_hidden_layers, num_node_features, hidden_channels, num_classes, readout, dropout,
                 batch_norm, residual=False):
        super(GraphSAGE, self).__init__()
        self.readout = readout
        self.batch_norm = batch_norm
        self.dropout = dropout

        if num_node_features != num_classes:
            self.residual = False
        if self.batch_norm:
            self.batchnorm_x = nn.BatchNorm1d(hidden_channels)
        if self.dropout:
            self.dropout = nn.Dropout(dropout)

        self.residual = residual
        self.input_layer = SAGEConv(num_node_features, hidden_channels)
        self.hidden_layers = nn.ModuleList(
            [SAGEConv(hidden_channels, hidden_channels) for _ in range(num_hidden_layers)])

        self.MLP_layer = MLPReadout(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.input_layer(x, edge_index)
        x = x.relu()

        for conv in self.hidden_layers:
            x_in = x
            x = conv(x, edge_index)
            if self.batch_norm:
                x = self.batchnorm_x(x)
            x = x.relu()
            if self.residual:
                x = x_in + x
            if self.dropout:
                x = self.dropout(x)
        if self.readout == 'max':
            x = global_max_pool(x, batch)
        elif self.readout == 'sum':
            x = global_add_pool(x, batch)
        elif self.readout == 'mean':
            x = global_mean_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)
        return self.MLP_layer(x)

    def predict(self, data, device):
        self.eval()
        data = data.to(device)
        out = self(data.x, data.edge_index, data.batch)
        score = F.softmax(out, dim=1)
        pred = score.argmax(dim=1)
        return pred


class GAT(nn.Module):
    def __init__(self, heads, num_hidden_layers, num_node_features, hidden_channels, num_classes, readout, dropout,
                 batch_norm,
                 residual=False):
        super(GAT, self).__init__()
        self.readout = readout
        self.batch_norm = batch_norm
        self.dropout = dropout

        if num_node_features != num_classes:
            self.residual = False
        if self.batch_norm:
            self.batchnorm_x = nn.BatchNorm1d(hidden_channels)
        if self.dropout:
            self.dropout = nn.Dropout(dropout)

        self.residual = residual
        self.input_layer = GATConv(num_node_features, hidden_channels // heads, heads=heads, concat=True)
        self.hidden_layers = nn.ModuleList(
            [GATConv(hidden_channels, hidden_channels // heads, heads=heads, concat=True) for _ in
             range(num_hidden_layers)])

        self.MLP_layer = MLPReadout(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.input_layer(x, edge_index)
        x = x.relu()
        for conv in self.hidden_layers:
            x_in = x
            x = conv(x, edge_index)
            if self.batch_norm:
                x = self.batchnorm_x(x)
            x = x.relu()
            if self.residual:
                x = x_in + x
            if self.dropout:
                x = self.dropout(x)
        if self.readout == 'max':
            x = global_max_pool(x, batch)
        elif self.readout == 'sum':
            x = global_add_pool(x, batch)
        elif self.readout == 'mean':
            x = global_mean_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)
        return self.MLP_layer(x)

    def predict(self, data, device):
        self.eval()
        data = data.to(device)
        out = self(data.x, data.edge_index, data.batch)
        score = F.softmax(out, dim=1)
        pred = score.argmax(dim=1)
        return pred
