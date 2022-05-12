import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import Linear, MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax

from src.definitions.SltEdgeTypes import SltEdgeTypes
from src.model.ScaledDotProductAttention import ScaledDotProductAttention


class AttBlock(MessagePassing):
    def __init__(self, device, f_size, in_size, out_size, init_size, dropout_p):
        super(AttBlock, self).__init__(node_dim=0, aggr='add')
        self.device = device
        self.dropout_p = dropout_p
        self.negative_slope = 0.2
        self.out_size = out_size

        self.lin_h = Linear(out_size, 128, bias=False, weight_initializer='glorot')
        self.lin_x_init = Linear(init_size, 128, bias=False, weight_initializer='glorot')
        self.lin_key = Linear(f_size, 128, bias=False, weight_initializer='glorot')
        self.lin_value = Linear(f_size, out_size, bias=False, weight_initializer='glorot')
        self.attention = ScaledDotProductAttention(dim=2)

    def reset_parameters(self):
        self.lin_h.reset_parameters()
        self.lin_x_init.reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_value.reset_parameters()

    def forward(self, f, x, h, edge_index, edge_type, f_batch, x_batch, x_init):
        h = self.lin_h(h)
        x_init = self.lin_x_init(x_init)
        c, alpha = self.propagate(edge_index, f=f, h=h, x_init=x_init, edge_type=edge_type, x_batch=x_batch, f_batch=f_batch)
        return c, alpha

    def message(self, h_i, x_init_j, edge_index, edge_type):
        pc_edges_mask = (edge_type == int(SltEdgeTypes.PARENT_CHILD)).to(torch.long).unsqueeze(1)
        bb_edges_mask = (edge_type == int(SltEdgeTypes.LEFTBROTHER_RIGHTBROTHER)).to(torch.long).unsqueeze(1)

        query_h = h_i * pc_edges_mask
        query_bro = x_init_j * bb_edges_mask
        query_pa = x_init_j * pc_edges_mask

        query = query_h + query_bro + query_pa
        return query

    def update(self, query, f, x_batch, f_batch):
        key = self.lin_key(f)
        value = self.lin_value(f)

        alpha_not_batch_mask = (x_batch.unsqueeze(1) - f_batch.unsqueeze(0) != 0).long()
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)
        context, attn = self.attention(query, key, value, alpha_not_batch_mask)
        context = context.squeeze(0)
        attn = attn.squeeze(0)

        # alpha_batch_mask = (x_batch.unsqueeze(1) - f_batch.unsqueeze(0) == 0).long()
        # score = torch.mm(query, key.transpose(1, 0))
        # attn = F.softmax(score.masked_fill((1 - alpha_batch_mask).bool(), float('-inf')), dim=1)
        # attn = F.dropout(attn, p=self.dropout_p, training=self.training)
        # context = torch.mm(attn, value)
        return context, attn

