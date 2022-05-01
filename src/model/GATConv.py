import torch
import torch.nn.functional as F
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros


class GATConv(MessagePassing):
    def __init__(
            self,
            in_size,
            out_size,
            edge_in_size,
            edge_out_size,
            heads=1,
            negative_slope=0.2,
            dropout=0.0,
            concat=False,
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_size = in_size
        self.out_size = out_size
        self.edge_in_size = edge_in_size
        self.edge_out_size = edge_out_size
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.concat = concat

        self.lin = Linear(in_size, heads * out_size, bias=False, weight_initializer='glorot')

        self.att_src = Parameter(torch.Tensor(1, heads, out_size))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_size))

        self.lin_transform_edge = Linear(2 * in_size + edge_in_size, edge_out_size, bias=False, weight_initializer='glorot')

        self.lin_edge = Linear(edge_out_size, heads * out_size, bias=False, weight_initializer='glorot')
        self.att_edge = Parameter(torch.Tensor(1, heads, out_size))

        if concat:
            self.bias = Parameter(torch.Tensor(heads * out_size))
        else:
            self.bias = Parameter(torch.Tensor(out_size))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.lin_transform_edge.reset_parameters()
        self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr=None, size=None):
        x_prev = x
        x = self.lin(x).view(-1, self.heads, self.out_size)

        alpha_src = (x * self.att_src).sum(dim=-1)
        alpha_dst = (x * self.att_dst).sum(dim=-1)
        alpha = (alpha_src, alpha_dst)

        num_nodes = x.size(0)
        edge_index, edge_attr = remove_self_loops(
            edge_index, edge_attr)
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr, fill_value='mean',
            num_nodes=num_nodes)

        # if element is set as tuple, the first will be used for source item (j),
        # the second for tgt item (i)
        alpha, edge_attr = self.edge_updater(edge_index, x_prev=x_prev, alpha=alpha, edge_attr=edge_attr)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        if self.concat:
            out = out.view(-1, self.heads * self.out_size)
        elif self.heads > 1:
            out = out.mean(dim=1)
        else:
            out = out.squeeze(1)
        out += self.bias
        return out, edge_index, edge_attr

    def edge_update(self, alpha_j, alpha_i, x_prev_j, x_prev_i, edge_attr, index, ptr, size_i):
        # j -> i
        edge_attr = self.lin_transform_edge(torch.cat((x_prev_i, edge_attr, x_prev_j), dim=1))
        edge_attr = F.leaky_relu(edge_attr, self.negative_slope)

        alpha = alpha_j + alpha_i

        edge_attr_mod = self.lin_edge(edge_attr)
        edge_attr_mod = edge_attr_mod.view(-1, self.heads, self.out_size)
        alpha_edge = (edge_attr_mod * self.att_edge).sum(dim=-1)
        alpha = alpha + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha, edge_attr

    def message(self, x_j, alpha):
        # j -> i
        return alpha.unsqueeze(-1) * x_j
