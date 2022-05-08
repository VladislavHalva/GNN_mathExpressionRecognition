import torch
from torch import nn
from torch.nn import Parameter
from torch_geometric.nn import Linear
from torch_geometric.nn.inits import zeros

from src.model.AttBlock import AttBlock
from src.model.GCN import GCN


class DecoderBlock(nn.Module):
    def __init__(self, device, f_size, in_size, out_size, att_dropout_p, is_first=False):
        super(DecoderBlock, self).__init__()
        self.device = device
        self.f_size = f_size
        self.in_size = in_size
        self.out_size = out_size
        self.is_first = is_first

        self.gcn = GCN(device, in_size, out_size, is_first)
        self.attBlock = AttBlock(device, f_size, in_size, out_size, att_dropout_p)

        self.lin = Linear(out_size, out_size, bias=False, weight_initializer='glorot')

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, f, x, edge_index, edge_type, f_batch, x_batch):
        h = self.gcn(x, edge_index, edge_type)
        c, alpha = self.attBlock(f, x, h, edge_index, edge_type, f_batch, x_batch)

        z = h + c
        z = self.lin(z)
        return z, alpha
