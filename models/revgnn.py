import os.path as osp
import time

import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch.nn import LayerNorm, Linear
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.nn import GroupAddRev, SAGEConv,GCNConv
from torch_geometric.utils import index_to_mask


class GNNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = LayerNorm(in_channels, elementwise_affine=True)
        # self.conv = SAGEConv(in_channels, out_channels)
        self.conv = GCNConv(out_channels, out_channels,normalize=False)

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, x, edge_index,dropout_mask=None,edge_weight=None):
        x = self.norm(x).relu()
        if self.training and dropout_mask is not None:
            x = x * dropout_mask
        return self.conv(x, edge_index, edge_weight)
        # return self.conv(x, edge_index)


class H_RevGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_groups=2):
        super(H_RevGNN, self).__init__()

        self.dropout = dropout

        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)
        self.norm = LayerNorm(hidden_channels, elementwise_affine=True)

        assert hidden_channels % num_groups == 0
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = GNNBlock(
                hidden_channels // num_groups,
                hidden_channels // num_groups,
            )
            ##TODO add edge_weight
            self.convs.append(GroupAddRev(conv, num_groups=num_groups))

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.norm.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self,data):
        x, edge_index = data.x, data.edge_index_graph
        x = self.lin1(x)

        # Generate a dropout mask which will be shared across GNN blocks:
        mask = None
        if self.training and self.dropout > 0:
            mask = torch.zeros_like(x).bernoulli_(1 - self.dropout)
            mask = mask.requires_grad_(False)
            mask = mask / (1 - self.dropout)
        mask=None

        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index, mask, data.edge_weight_graph)
        x = self.norm(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)