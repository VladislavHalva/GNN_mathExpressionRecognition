from torch import nn
from torch_geometric.utils import remove_self_loops

from src.model.GATConv import GATConv
from src.model.VGG import VGG


class Encoder(nn.Module):
    def __init__(self, edge_features, edge_h_size, in_size, h_size, out_size):
        super(Encoder, self).__init__()

        self.vgg = VGG(1, in_size, dropout_p=0.2)

        self.gat1 = GATConv(in_size, h_size, edge_features, edge_h_size)
        self.gat2 = GATConv(h_size, h_size, edge_h_size, edge_h_size)
        self.gat3 = GATConv(h_size, out_size, edge_h_size, edge_h_size)

    def forward(self, x, edge_index, edge_attr):
        # extract component level visual features
        x = self.vgg(x)

        x, edge_index, edge_attr = self.gat1(x, edge_index, edge_attr)
        x, edge_index, edge_attr = self.gat2(x, edge_index, edge_attr)
        x, edge_index, edge_attr = self.gat3(x, edge_index, edge_attr)

        # remove self loops added during GAT layers processing
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        return x, edge_index, edge_attr
