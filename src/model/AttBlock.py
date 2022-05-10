import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import Linear

from src.definitions.SltEdgeTypes import SltEdgeTypes
from src.model.ParentGetter import ParentGetter


class AttBlock(nn.Module):
    def __init__(self, device, f_size, in_size, out_size, init_size, dropout_p):
        super(AttBlock, self).__init__()
        self.device = device
        self.dropout_p = dropout_p

        self.pa_identifier = ParentGetter(device)
        self.bro_identifier = ParentGetter(device)

        self.lin_h = Linear(out_size, init_size, bias=False, weight_initializer='glorot')

        self.lin_alpha = Linear(f_size, init_size, bias=False, weight_initializer='glorot')
        self.lin_f = Linear(f_size, out_size, bias=False, weight_initializer='glorot')

        self.lin_out = Linear(out_size, out_size, bias=False, weight_initializer='glorot')

    def reset_parameters(self):
        self.lin_h.reset_parameters()
        self.lin_alpha.reset_parameters()
        self.lin_f.reset_parameters()
        self.lin_out.reset_parameters()

    def forward(self, f, x, h, edge_index, edge_type, f_batch, x_batch, x_init):
        pc_edges_mask = (edge_type == SltEdgeTypes.PARENT_CHILD).to(torch.long).to(self.device)
        bb_edges_mask = (edge_type == SltEdgeTypes.LEFTBROTHER_RIGHTBROTHER).to(torch.long).to(self.device)

        pc_edge_index = edge_index.t()[pc_edges_mask.nonzero(as_tuple=False).view(-1)].t().contiguous()
        bb_edge_index = edge_index.t()[bb_edges_mask.nonzero(as_tuple=False).view(-1)].t().contiguous()

        pa = self.pa_identifier(x_init, pc_edge_index)
        bro = self.bro_identifier(x_init, bb_edge_index)

        q = self.lin_h(h) + pa + bro
        alpha = q @ self.lin_alpha(f).t().contiguous()

        alpha_batch_mask = (x_batch.unsqueeze(1) - f_batch.unsqueeze(0) == 0).long().to(self.device)
        alpha = alpha * alpha_batch_mask / (torch.sum(torch.abs(alpha * alpha_batch_mask), dim=1) + 0.0001).unsqueeze(1)
        # alpha = F.softmax(alpha.masked_fill((1 - alpha_batch_mask).bool(), float('-inf')), dim=1)
        # alpha = F.dropout(alpha, p=self.dropout_p, training=self.training)
        c = alpha @ self.lin_f(f)
        # print(alpha[0])
        return c, alpha
