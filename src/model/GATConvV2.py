# ###
# Mathematical expression recognition tool.
# Written as a part of masters thesis at VUT FIT Brno, 2022

# Author: Vladislav Halva
# Login: xhalva04
# Derived from Pytorch Geometrics GATConvV2 implementation https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GATv2Conv
# ###

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros


class GATConvV2(MessagePassing):
    def __init__(self, in_size, out_size, edge_in_size, edge_out_size, heads=1,
                 negative_slope=0.2, dropout=0.0, concat=False):
        """
        :param in_size: in node features size
        :param out_size: out node features size
        :param edge_in_size: in edge features size
        :param edge_out_size: out edge features size
        :param heads: number of attention heads
        :param negative_slope: leakyReLU negative slope
        :param dropout: dropout probability
        :param concat: if True concatenate heads outputs, else mean
        """
        super().__init__(node_dim=0, aggr='add')
        self.in_size = in_size
        self.out_size = out_size
        self.edge_in_size = edge_in_size
        self.edge_out_size = edge_out_size
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.concat = concat

        # nodes transformation
        self.lin_src = Linear(in_size, heads * out_size, bias=False, weight_initializer='glorot')
        self.lin_tgt = Linear(in_size, heads * out_size, bias=False, weight_initializer='glorot')
        # nodes attention contribution
        self.att = Parameter(torch.Tensor(1, heads, out_size))
        # edges transformation
        self.lin_transform_edge = Linear(2 * in_size + edge_in_size, edge_out_size, bias=False, weight_initializer='glorot')
        # edges transformation for attention contribution onlu
        self.lin_edge = Linear(edge_out_size, heads * out_size, bias=False, weight_initializer='glorot')

        if concat:
            self.bias = Parameter(torch.Tensor(heads * out_size))
        else:
            self.bias = Parameter(torch.Tensor(out_size))
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_tgt.reset_parameters()
        self.lin_transform_edge.reset_parameters()
        self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr=None, size=None):
        # store previous node features for edges update
        x_prev = x
        # nodes transformation
        x_src = self.lin_src(x).view(-1, self.heads, self.out_size)
        x_tgt = self.lin_tgt(x).view(-1, self.heads, self.out_size)
        # nodes attn contribution (src a dst separately) -> needs to emulate concatenation
        # add self-loops
        num_nodes = x.size(0)
        edge_index, edge_attr = remove_self_loops(
            edge_index, edge_attr)
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr, fill_value='mean',
            num_nodes=num_nodes)
        # update edge features and compute attention
        edge_attr = self.edge_updater(edge_index, x_prev=x_prev, edge_attr=edge_attr)
        # update node features
        out = self.propagate(edge_index, x=(x_src, x_tgt), edge_attr=edge_attr, size=size)
        # transform node features from various heads based on preferences
        if self.concat:
            out = out.view(-1, self.heads * self.out_size)
        elif self.heads > 1:
            out = out.mean(dim=1)
        else:
            out = out.squeeze(1)
        out += self.bias
        return out, edge_index, edge_attr

    def edge_update(self, x_prev_j, x_prev_i, edge_attr):
        # j -> i
        # update edge features based on connected nodes features and prev. edge features
        edge_attr = self.lin_transform_edge(torch.cat((x_prev_i, edge_attr, x_prev_j), dim=1))
        edge_attr = F.leaky_relu(edge_attr, self.negative_slope)
        return edge_attr

    def message(self, x_j, x_i, edge_attr, index, ptr, size_i):
        # j -> i
        x = x_i + x_j
        edge_attr = self.lin_edge(edge_attr)
        edge_attr = edge_attr.view(-1, self.heads, self.out_size)
        x *= torch.sigmoid(edge_attr)
        # x = x + edge_attr
        # attend
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

