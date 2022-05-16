# ###
# Mathematical expression recognition tool.
# Written as a part of masters thesis at VUT FIT Brno, 2022

# Author: Vladislav Halva
# Login: xhalva04
# ###

from torch import nn
from torch_geometric.utils import remove_self_loops

from src.model.GATConvV2 import GATConvV2
from src.model.VGG11 import VGG11


class Encoder(nn.Module):
    def __init__(self, edge_features, edge_h_size, in_size, h_size, out_size, vocab_size, vgg_dropout_p, gat_dropout_p):
        """
        :param edge_features: input edge features size
        :param edge_h_size: hidden and output edge features size
        :param in_size: input node features size (output size of VGG)
        :param h_size: hidden node feature size
        :param out_size: output node features size
        :param vocab_size: vocabulary size/length
        :param vgg_dropout_p: dropout probability for vgg linear layers
        :param gat_dropout_p: dropout probability for GAT layers
        """
        super(Encoder, self).__init__()

        self.vgg = VGG11(1, in_size, dropout_p=vgg_dropout_p)

        self.gat1 = GATConvV2(in_size, h_size, edge_features, edge_h_size, dropout=gat_dropout_p, heads=3)
        self.gat2 = GATConvV2(h_size, h_size, edge_h_size, edge_h_size, dropout=gat_dropout_p, heads=3)
        self.gat3 = GATConvV2(h_size, out_size, edge_h_size, edge_h_size, dropout=gat_dropout_p, heads=3)

        self.lin_x_conv_out = nn.Linear(in_size, vocab_size, bias=False)

    def forward(self, x, edge_index, edge_attr):
        # extract component level visual features
        x = self.vgg(x)
        # node classification logits after feature extraction
        x_conv_score = self.lin_x_conv_out(x)
        # process with GAT layers
        x, edge_index, edge_attr = self.gat1(x, edge_index, edge_attr)
        x, edge_index, edge_attr = self.gat2(x, edge_index, edge_attr)
        x, edge_index, edge_attr = self.gat3(x, edge_index, edge_attr)
        # remove self loops added during GAT layers processing
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        return x, edge_index, edge_attr, x_conv_score
