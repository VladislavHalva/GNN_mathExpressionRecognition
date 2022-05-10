import torch_geometric.nn
import torchvision.models
from torch import nn
from torch_geometric.utils import remove_self_loops

from src.model.GATConv import GATConv
from src.model.VGG import VGG


class Encoder(nn.Module):
    def __init__(self, edge_features, edge_h_size, in_size, h_size, out_size, vocab_size, vgg_dropout_p, gat_dropout_p):
        super(Encoder, self).__init__()

        self.vgg = VGG(1, in_size, dropout_p=vgg_dropout_p)
        # self.vgg = torchvision.models.vgg13(pretrained=True)
        # self.vgg.classifier[6] = nn.Linear(4096, in_size)

        self.gat1 = GATConv(in_size, h_size, edge_features, edge_h_size, dropout=gat_dropout_p, heads=3)
        self.gat2 = GATConv(h_size, h_size, edge_h_size, edge_h_size, dropout=gat_dropout_p, heads=3)
        self.gat3 = GATConv(h_size, out_size, edge_h_size, edge_h_size, dropout=gat_dropout_p, heads=3)

        # self.gat1 = GATConv(in_size, out_size, edge_features, edge_h_size, dropout=gat_dropout_p)

        self.lin_comp_out = nn.Linear(in_size, vocab_size, bias=False)

    def forward(self, x, edge_index, edge_attr):
        # extract component level visual features
        # x = x.repeat(1, 3, 1, 1)
        x = self.vgg(x)
        comp_class = self.lin_comp_out(x)

        x, edge_index, edge_attr = self.gat1(x, edge_index, edge_attr)
        x, edge_index, edge_attr = self.gat2(x, edge_index, edge_attr)
        x, edge_index, edge_attr = self.gat3(x, edge_index, edge_attr)

        # remove self loops added during GAT layers processing
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        return x, edge_index, edge_attr, comp_class
