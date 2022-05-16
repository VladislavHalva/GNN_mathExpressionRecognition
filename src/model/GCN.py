# ###
# Mathematical expression recognition tool.
# Written as a part of masters thesis at VUT FIT Brno, 2022

# Author: Vladislav Halva
# Login: xhalva04
# ###

import torch
from torch_geometric.nn import MessagePassing, Linear
import torch.nn.functional as F

from src.definitions.SltEdgeTypes import SltEdgeTypes


class GCN(MessagePassing):
    """
    Modified GCN layer with multiple weight matrices for each edge type.
    """
    def __init__(self, device, in_size, out_size, is_first=False):
        """
        :param device: device
        :param in_size: in node features size
        :param out_size: out node features size
        :param is_first: True if within first decoder block
        """
        super(GCN, self).__init__(node_dim=0, aggr='add')
        self.device = device
        self.in_size = in_size
        self.out_size = out_size
        self.is_first = is_first

        self.lin_gg = Linear(in_size, out_size, bias=False, weight_initializer='glorot')
        self.lin_pc = Linear(in_size, out_size, bias=False, weight_initializer='glorot')
        self.lin_bb = Linear(in_size, out_size, bias=False, weight_initializer='glorot')
        self.lin_cc = Linear(in_size, out_size, bias=False, weight_initializer='glorot')

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_gg.reset_parameters()
        self.lin_pc.reset_parameters()
        self.lin_bb.reset_parameters()
        self.lin_cc.reset_parameters()

    def forward(self, x, edge_index, edge_type):
        """
        :param x: node features
        :param edge_index: edge index
        :param edge_type: edge types list
        :return: new node features
        """
        # get transformations for each node as if it was connected by each of the types of edges
        gg = self.lin_gg(x)
        pc = self.lin_pc(x)
        bb = self.lin_bb(x)

        if self.is_first:
            # if first layer - self features are zero for each embedding
            # important to simulate evaluation time state during training
            cc = self.lin_cc(torch.zeros(x.size(), dtype=torch.double).to(self.device))
        else:
            cc = self.lin_cc(x)

        h = self.propagate(gg=gg, pc=pc, bb=bb, cc=cc, edge_index=edge_index, edge_type=edge_type)
        h_in_and_condition = torch.cat([h, h], dim=1)
        h = F.glu(h_in_and_condition, dim=1)
        return h

    def message(self, gg_j, pc_j, bb_j, cc_j, edge_index, edge_type):
        """
        :param gg_j: source node features as if it was connected by grandparent edge
        :param pc_j: source node features as if it was connected by parent edge
        :param bb_j: source node features as if it was connected by brother edge
        :param cc_j: source node features as if it was connected by self-loop edge
        :param edge_index: edge index
        :param edge_type: edge types list
        :return: message to h_i
        """
        # j -> i
        gg_edge_mask = (edge_type == SltEdgeTypes.GRANDPARENT_GRANDCHILD).to(torch.long).unsqueeze(1).to(self.device)
        pc_edge_mask = (edge_type == SltEdgeTypes.PARENT_CHILD).to(torch.long).unsqueeze(1).to(self.device)
        bb_edge_mask = (edge_type == SltEdgeTypes.LEFTBROTHER_RIGHTBROTHER).to(torch.long).unsqueeze(1).to(self.device)
        cc_edge_mask = (edge_type == SltEdgeTypes.CURRENT_CURRENT).to(torch.long).unsqueeze(1).to(self.device)
        # mask messages based on edge types
        gg_j = gg_j * gg_edge_mask
        pc_j = pc_j * pc_edge_mask
        bb_j = bb_j * bb_edge_mask
        cc_j = cc_j * cc_edge_mask

        h = gg_j + pc_j + bb_j + cc_j
        return h
