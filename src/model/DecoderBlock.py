# ###
# Mathematical expression recognition tool.
# Written as a part of masters thesis at VUT FIT Brno, 2022

# Author: Vladislav Halva
# Login: xhalva04
# ###

from torch import nn
from torch_geometric.nn import Linear

from src.model.AttentionBlock import AttentionBlock
from src.model.GCN import GCN


class DecoderBlock(nn.Module):
    """
    Decoder layer containing modified GCN and attention to source graph.
    """
    def __init__(self, device, f_size, in_size, out_size, att_size, att_dropout_p, init_size, is_first=False):
        """
        :param device: device
        :param f_size: source graph node features size
        :param in_size: output graph in node features size
        :param out_size: output graph out node features size
        :param att_size: attention vector size
        :param att_dropout_p: dropout probability
        :param init_size: output graph initial node embedding size
        :param is_first: True if layer is first in decoder
        """
        super(DecoderBlock, self).__init__()
        self.device = device
        self.f_size = f_size
        self.in_size = in_size
        self.out_size = out_size
        self.init_size = init_size
        self.is_first = is_first

        self.gcn = GCN(device, in_size, out_size, is_first)
        self.attBlock = AttentionBlock(device, f_size, out_size, init_size, att_size, att_dropout_p, is_first=is_first)
        self.lin = Linear(out_size, out_size, bias=True, weight_initializer='glorot')

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, f, x, edge_index, edge_type, f_batch, x_batch, x_init):
        """
        :param f: source graph node features
        :param x: output graph node features
        :param edge_index: output graph edge index
        :param edge_type: output graph edge types
        :param f_batch: source graph nodes batch indices
        :param x_batch: output graph nodes batch indices
        :param x_init: output graph initial node embeddings
        :return: output graph new node features, attention alpha coefficients
        """
        # process output graph with GCN
        h = self.gcn(x, edge_index, edge_type)
        # get context vector from source graph
        c, alpha = self.attBlock(f, h, edge_index, edge_type, f_batch, x_batch, x_init)
        # combine
        z = h + c
        z = self.lin(z)
        return z, alpha
