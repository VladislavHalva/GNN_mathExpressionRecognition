import torch
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing, Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, softmax, remove_self_loops
import torch.nn.functional as F


class GATLayer(MessagePassing):
    def __init__(self, input_size, hidden_size):
        super(GATLayer, self).__init__(node_dim=0, aggr='add')
        # node_dim = axis along which propagation is done
        # aggr = aggregation function (add = SUM)

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lin_edge_update = Linear(3 * input_size, input_size, bias=False, weight_initializer='glorot')

        self.lin = Linear(input_size, hidden_size, bias=False, weight_initializer='glorot')
        self.att = Parameter(torch.Tensor(1, hidden_size))

        self.lin_edge = Linear(input_size, hidden_size, bias=False, weight_initializer='glorot')
        self.att_edge = Parameter(torch.Tensor(1, hidden_size))

        self.bias = Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.lin_edge.reset_parameters()
        self.lin_edge_update.reset_parameters()
        glorot(self.att)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr, size=None):
        # remove self loop if they were added in previous layer
        edge_index, edge_attr = remove_self_loops(
            edge_index, edge_attr)
        # add self loops
        edge_index, edge_attr = add_self_loops(
            edge_index, edge_attr,
            fill_value=torch.zeros(self.input_size, dtype=float),
            num_nodes=x.size(0)
        )
        # update edge features
        edge_attr = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)
        # transform input features
        x = self.lin(x)
        # compute node-level attention for input nodes
        alpha = (x * self.att).sum(dim=-1)
        # compute edge attributes lin. transformation
        edge_attr = self.lin_edge(edge_attr)
        # propagation
        out = self.propagate(edge_index, x=x, alpha=alpha, edge_attr=edge_attr, size=size)
        return out, edge_index, edge_attr

    def edge_update(self, x_i, x_j, edge_attr):
        concatenated_features = torch.cat((x_i, edge_attr, x_j), dim=-1)
        return F.leaky_relu(self.lin_edge_update(concatenated_features))

    def message(self, x_j, alpha_j, alpha_i, edge_attr, index, ptr, size_i):
        # sum attention contributions of src and tgt edge node features
        alpha = alpha_i + alpha_j
        alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)

        # add attention contribution of edge features
        alpha = alpha + alpha_edge

        # compute node feature update
        alpha = F.leaky_relu(alpha, 0.1)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=0.1, training=self.training)
        return F.leaky_relu(x_j * alpha.unsqueeze(-1), 0.1)
