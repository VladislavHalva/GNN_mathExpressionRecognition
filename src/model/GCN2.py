# ###
# Mathematical expression recognition tool.
# Written as a part of masters thesis at VUT FIT Brno, 2022

# Author: Vladislav Halva
# Login: xhalva04
# ###

import torch
from torch.nn import ModuleList
from torch_geometric.nn import MessagePassing, Linear
import torch.nn.functional as F

from src.definitions.SltEdgeTypes import SltEdgeTypes


class GCN2(MessagePassing):
    def __init__(self, device, in_size, out_size, is_first=False):
        super(GCN2, self).__init__(node_dim=0, aggr='add')
        self.device = device
        self.in_size = in_size
        self.out_size = out_size
        self.is_first = is_first

        self.lins_l = ModuleList([
            Linear(in_size, out_size, bias=True, weight_initializer='glorot'),
            Linear(in_size, out_size, bias=True, weight_initializer='glorot'),
            Linear(in_size, out_size, bias=True, weight_initializer='glorot'),
            Linear(in_size, out_size, bias=True, weight_initializer='glorot')
        ])

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins_l:
            lin.reset_parameters()

    def forward(self, x, edge_index, edge_type):
        h = self.propagate(x=x, edge_index=edge_index, edge_type=edge_type)
        h_in_and_condition = torch.cat([h, h], dim=1)
        h = F.glu(h_in_and_condition, dim=1)
        return h

    def message(self, x_j, edge_index, edge_type):
        pc_indices = ((edge_type == int(SltEdgeTypes.PARENT_CHILD)).nonzero(as_tuple=True)[0])
        gg_indices = ((edge_type == int(SltEdgeTypes.GRANDPARENT_GRANDCHILD)).nonzero(as_tuple=True)[0])
        bb_indices = ((edge_type == int(SltEdgeTypes.LEFTBROTHER_RIGHTBROTHER)).nonzero(as_tuple=True)[0])
        cc_indices = ((edge_type == int(SltEdgeTypes.CURRENT_CURRENT)).nonzero(as_tuple=True)[0])

        x_j[pc_indices] = self.lins_l[int(SltEdgeTypes.PARENT_CHILD)](x_j[pc_indices])
        x_j[gg_indices] = self.lins_l[int(SltEdgeTypes.GRANDPARENT_GRANDCHILD)](x_j[gg_indices])
        x_j[bb_indices] = self.lins_l[int(SltEdgeTypes.LEFTBROTHER_RIGHTBROTHER)](x_j[bb_indices])
        if self.is_first:
            x_cc_init = torch.zeros((cc_indices.shape[0], self.in_size), dtype=torch.double).to(self.device)
            x_j[cc_indices] = self.lins_l[int(SltEdgeTypes.CURRENT_CURRENT)](x_cc_init)
        else:
            x_j[cc_indices] = self.lins_l[int(SltEdgeTypes.CURRENT_CURRENT)](x_j[cc_indices])

        return x_j
