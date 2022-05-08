import torch
from torch_geometric.nn import MessagePassing


class ParentGetter(MessagePassing):
    def __init__(self, device):
        super(ParentGetter, self).__init__(node_dim=0, aggr='add')
        self.device = device

    def forward(self, x, edge_index):
        out = torch.zeros(x.shape, dtype=torch.float).to(self.device)
        out = self.propagate(x=x, out=out, edge_index=edge_index)
        return out

    def message(self, x_j, edge_index):
        return x_j

    def update(self, x, out):
        return x + out
