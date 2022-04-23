import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import Linear
from torch_geometric.nn import GATConv
from torch_geometric.utils import remove_self_loops
import torchvision.models as models
import cv2 as cv
import torch.nn.functional as F

from src.model.GATlayer import GATLayer
from src.model.Resnet import Resnet
from src.model.VGG16 import VGG16


class Encoder(nn.Module):
    def __init__(self, device, components_shape, edge_features, in_size, h_size, out_size):
        super(Encoder, self).__init__()

        # OPTION 1 - use resnet for feature extraction
        # self.resnet = Resnet(1, in_size, components_shape)

        # OPTION 2 - use pretrained vgg for feature extraction
        # self.vgg = models.vgg16(pretrained=True)
        # modify last layer to fit the desired input feature size for GAT
        # self.vgg.classifier[6] = nn.Linear(4096, in_size)
        # for param in self.vgg.features.parameters():
        #     param.requires_grad = False
        # for param in self.vgg.classifier.parameters():
        #     param.requires_grad = True
        self.vgg = VGG16(1, in_size)

        # self.lin_edge = nn.Sequential(
        #     Linear(edge_features, 100, bias=False),
        #     nn.ReLU())
        # self.gat1 = GATLayer(in_size, h_size)
        # self.gat2 = GATLayer(h_size, h_size)
        # self.gat3 = GATLayer(h_size, out_size)

        self.gatPYG1 = GATConv(in_size, h_size, 1, edge_dim=edge_features)
        self.gatPYG2 = GATConv(h_size, h_size, 1, edge_dim=edge_features)
        self.gatPYG3 = GATConv(h_size, out_size, 1, edge_dim=edge_features)

    def forward(self, x, edge_index, edge_attr):
        # extract component level visual features
        # OPTION 1 : resnet
        # x = self.resnet(x)

        # OPTION 2 : vgg-16 with grayscale expanded to 3 channels
        # x = x.expand(-1, 3, -1, -1)
        x = self.vgg(x)

        # project edge features to node features dimensionality
        # edge_attr = F.leaky_relu(self.lin_edge(edge_attr), negative_slope=0.1)

        # pass data through 3 layer GAT network
        # x, edge_index, edge_attr = self.gat1(x, edge_index, edge_attr)
        # x, edge_index, edge_attr = self.gat2(x, edge_index, edge_attr)
        # x, edge_index, edge_attr = self.gat3(x, edge_index, edge_attr)

        x = self.gatPYG1(x, edge_index, edge_attr)
        x = self.gatPYG2(x, edge_index, edge_attr)
        x = self.gatPYG3(x, edge_index, edge_attr)

        # remove self loops added during GAT layers processing
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        return x, edge_index, edge_attr
