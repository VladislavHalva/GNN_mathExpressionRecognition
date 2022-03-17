from torch import nn
from torch.nn import Linear
from torch_geometric.utils import remove_self_loops

from src.model.GATlayer import GATLayer
from src.model.Resnet import Resnet


class Encoder(nn.Module):
    def __init__(self, device, components_shape, input_edge_size, input_feature_size, hidden_size, embed_size):
        super(Encoder, self).__init__()
        self.resnet = Resnet(1, input_feature_size, components_shape)
        self.lin_edge = nn.Sequential(
            Linear(input_edge_size, input_feature_size, bias=False),
            nn.ReLU())
        self.gat1 = GATLayer(input_feature_size, hidden_size)
        self.gat2 = GATLayer(hidden_size, embed_size)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # extract component level visual features
        x = self.resnet(x)
        # project edge features to node features dimensionality
        edge_attr = self.lin_edge(edge_attr)

        # pass data through 2 layer GAT network
        x, edge_index, edge_attr = self.gat1(x, edge_index, edge_attr)
        x, edge_index, edge_attr = self.gat2(x, edge_index, edge_attr)

        # remove self loops added during GAT layers processing
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

        data.x = x
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        return data
