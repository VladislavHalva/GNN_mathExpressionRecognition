# ###
# Mathematical expression recognition tool.
# Written as a part of masters thesis at VUT FIT Brno, 2022

# Author: Vladislav Halva
# Login: xhalva04
# ###

import torch
from torch.nn import Parameter
from torch_geometric.nn import Linear, MessagePassing
from torch_geometric.nn.inits import glorot

from src.definitions.SltEdgeTypes import SltEdgeTypes
from src.model.Attention import Attention


class AttentionBlock(MessagePassing):
    """
    Decoder block source graph attention block.
    """
    def __init__(self, device, f_size, out_size, init_size, att_size, dropout_p, is_first=False):
        """
        :param device: device
        :param f_size: source graph node features size
        :param out_size: decoder layer output node feature size
        :param init_size: decoder initial node embedding size
        :param att_size: attention vector size
        :param dropout_p: dropout probability
        :param is_first: True if within first decoder block
        """
        super(AttentionBlock, self).__init__(node_dim=0, aggr='add')
        self.device = device
        self.dropout_p = dropout_p
        self.negative_slope = 0.2
        self.att_size = att_size
        self.out_size = out_size
        self.is_first = is_first

        self.weight = Parameter(
            torch.Tensor(3, out_size, att_size))

        self.lin_h = Linear(out_size, att_size, bias=False, weight_initializer='glorot')
        self.lin_x_init = Linear(init_size, att_size, bias=False, weight_initializer='glorot')
        self.lin_key = Linear(f_size, att_size, bias=False, weight_initializer='glorot')
        self.lin_value = Linear(f_size, out_size, bias=False, weight_initializer='glorot')
        self.attention = Attention(dropout_p=dropout_p)

    def reset_parameters(self):
        self.lin_h.reset_parameters()
        self.lin_x_init.reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_value.reset_parameters()
        glorot(self.weight)

    def forward(self, f, h, edge_index, edge_type, f_batch, x_batch, x_init):
        """
        :param f: source graph node features
        :param h: output graph node feature after GCN
        :param edge_index: edge index
        :param edge_type: edge types
        :param f_batch: source graph nodes batch indices
        :param x_batch: output graph nodes batch indices
        :param x_init: output graph initial node embeddings
        :return: context vectors for each output graph nodes, alpha coefficients
        """
        size = (h.size(0), h.size(0))
        query = torch.zeros(h.size(0), self.att_size, device=self.device)

        # propagate for each edge type separately
        for i, i_type in enumerate([SltEdgeTypes.CURRENT_CURRENT, SltEdgeTypes.PARENT_CHILD, SltEdgeTypes.LEFTBROTHER_RIGHTBROTHER]):
            edge_index_masked = self.masked_edge_index(edge_index, edge_type == i_type)
            if i_type == SltEdgeTypes.CURRENT_CURRENT:
                query_i = self.propagate(edge_index_masked, h=h, size=size)
            else:
                query_i = self.propagate(edge_index_masked, h=x_init, size=size)
            query = query + (query_i @ self.weight[i])

        key = self.lin_key(f)
        value = self.lin_value(f)

        alpha_not_batch_mask = (x_batch.unsqueeze(1) - f_batch.unsqueeze(0) != 0).long()
        context, attn = self.attention(query, key, value, alpha_not_batch_mask)
        return context, attn

    def message(self, h_j):
        """
        Only passes source node features for each edge of the currently processed type.
        :param h_j: source node features
        """
        return h_j

    def masked_edge_index(self, edge_index, edge_mask):
        """
        Mask edge index. Return edges given by mask.
        """
        return edge_index[:, edge_mask]
