import torch
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing, Linear
from torch_geometric.nn.inits import zeros
import torch.nn.functional as F

from src.definitions.SltEdgeTypes import SltEdgeTypes


class GCN(MessagePassing):
    def __init__(self, device, in_size, out_size, is_first=False):
        super(GCN, self).__init__(node_dim=0, aggr='add')
        self.device = device
        self.in_size = in_size
        self.out_size = out_size
        self.is_first = is_first

        self.lin_gg = Linear(in_size, out_size, bias=False, weight_initializer='glorot')
        self.lin_pc = Linear(in_size, out_size, bias=False, weight_initializer='glorot')
        self.lin_bb = Linear(in_size, out_size, bias=False, weight_initializer='glorot')
        self.lin_cc = Linear(in_size, out_size, bias=False, weight_initializer='glorot')

        # self.bias = Parameter(torch.Tensor(1, out_size))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_gg.reset_parameters()
        self.lin_pc.reset_parameters()
        self.lin_bb.reset_parameters()
        self.lin_cc.reset_parameters()
        # zeros(self.bias)

    def forward(self, x, edge_index, edge_type):
        gg = self.lin_gg(x)
        pc = self.lin_pc(x)
        bb = self.lin_bb(x)

        if self.is_first:
            cc = self.lin_cc(torch.zeros(x.size(), dtype=torch.float).to(self.device))
        else:
            cc = self.lin_cc(x)

        out = self.propagate(gg=gg, pc=pc, bb=bb, cc=cc, edge_index=edge_index, edge_type=edge_type)
        # out = out + self.bias
        return out

    def message(self, gg_j, pc_j, bb_j, cc_j, edge_index, edge_type):
        gg_edge_mask = (edge_type == SltEdgeTypes.GRANDPARENT_GRANDCHILD).to(torch.long).unsqueeze(1).to(self.device)
        pc_edge_mask = (edge_type == SltEdgeTypes.PARENT_CHILD).to(torch.long).unsqueeze(1).to(self.device)
        bb_edge_mask = (edge_type == SltEdgeTypes.LEFTBROTHER_RIGHTBROTHER).to(torch.long).unsqueeze(1).to(self.device)
        cc_edge_mask = (edge_type == SltEdgeTypes.CURRENT_CURRENT).to(torch.long).unsqueeze(1).to(self.device)

        gg_j = gg_j * gg_edge_mask
        pc_j = pc_j * pc_edge_mask
        bb_j = bb_j * bb_edge_mask
        cc_j = cc_j * cc_edge_mask

        h = torch.sum(torch.stack([gg_j, pc_j, bb_j, cc_j]), dim=0)
        h_in_and_condition = torch.cat([h, h], dim=1)
        h = F.glu(h_in_and_condition, dim=1)
        # h = F.relu(h)
        return h
