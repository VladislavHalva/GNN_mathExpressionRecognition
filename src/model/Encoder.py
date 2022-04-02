from torch import nn
from torch.nn import Linear
from torch_geometric.utils import remove_self_loops

from src.model.GATlayer import GATLayer
from src.model.Resnet import Resnet

import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, device, components_shape, input_edge_size, input_feature_size, hidden_size, embed_size):
        super(Encoder, self).__init__()

        # OPTION 1 - use resnet for feature extraction
        # self.resnet = Resnet(1, input_feature_size, components_shape)

        # OPTION 2 - use pretrained vgg for feature extraction
        self.vgg = models.vgg16(pretrained=True, progress=True, num_classes=1000)
        # modify last layer to fit the desired input feature size for GAT
        self.vgg.classifier[6] = nn.Linear(4096, input_feature_size)

        self.lin_edge = nn.Sequential(
            Linear(input_edge_size, input_feature_size, bias=False),
            nn.ReLU())
        self.gat1 = GATLayer(input_feature_size, hidden_size)
        self.gat2 = GATLayer(hidden_size, hidden_size)
        self.gat3 = GATLayer(hidden_size, embed_size)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # extract component level visual features

        # OPTION 1 : resnet
        # x = self.resnet(x)

        # OPTION 2 : vgg-16 with grayscale expanded to 3 channels
        x = x.expand(-1, 3, -1, -1)
        x = self.vgg(x)

        # project edge features to node features dimensionality
        edge_attr = self.lin_edge(edge_attr)

        # pass data through 3 layer GAT network
        x, edge_index, edge_attr = self.gat1(x, edge_index, edge_attr)
        x, edge_index, edge_attr = self.gat2(x, edge_index, edge_attr)
        x, edge_index, edge_attr = self.gat3(x, edge_index, edge_attr)

        # remove self loops added during GAT layers processing
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

        data.x = x
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        return data
