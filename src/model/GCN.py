# ###
# Mathematical expression recognition tool.
# Written as a part of masters thesis at VUT FIT Brno, 2022

# Author: Vladislav Halva
# Login: xhalva04
# ###

import torch
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros

from src.definitions.SltEdgeTypes import SltEdgeTypes


class GCN(MessagePassing):
    """
    Modified GCN layer with separate parameters for each edge type.
    """
    def __init__(self, device, in_size, out_size, is_first=False):
        """
        :param device: device
        :param in_size: in node features size
        :param out_size: out node features size
        :param is_first: True if within first decoder block
        """
        super(GCN, self).__init__(node_dim=0, aggr='mean')
        self.device = device
        self.in_size = in_size
        self.out_size = out_size
        self.num_edge_types = 4
        self.is_first = is_first

        # weight matrices for separate edge types
        self.weight = Parameter(
            torch.Tensor(self.num_edge_types, in_size, out_size))
        self.bias = Parameter(torch.Tensor(out_size))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_type):
        """
        :param x: node features
        :param edge_index: edge index
        :param edge_type: edge types list
        :return: new node features
        """
        size = (x.size(0), x.size(0))
        out = torch.zeros(x.size(0), self.out_size, device=self.device)

        # propagate for each edge type separately
        for i in range(self.num_edge_types):
            edge_index_masked = self.masked_edge_index(edge_index, edge_type == i)
            if self.is_first and i == SltEdgeTypes.CURRENT_CURRENT:
                # for training time self feature need to be zero for each node to be consistent with eval-time
                x_in = torch.zeros(x.size(0), self.in_size, dtype=torch.double, device=self.device)
            else:
                x_in = x
            h = self.propagate(edge_index_masked, x=x_in, size=size)
            out = out + (h @ self.weight[i])

        out += self.bias
        return out

    def message(self, x_j):
        """
        Only passes source node features for each edge of the currently processed type.
        :param x_j: source node features
        """
        return x_j

    def message_and_aggregate(self, adj_t, x):
        """
        Aggregation of neighborhood.
        :param adj_t: adjacency matrix
        :param x: node features
        :return: new node features per node
        """
        adj_t = adj_t.set_value(None)
        return torch.matmul(adj_t, x, reduce=self.aggr)

    def masked_edge_index(self, edge_index, edge_mask):
        """
        Mask edge index. Return edges given by mask.
        """
        return edge_index[:, edge_mask]
