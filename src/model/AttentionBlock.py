# ###
# Mathematical expression recognition tool.
# Written as a part of masters thesis at VUT FIT Brno, 2022

# Author: Vladislav Halva
# Login: xhalva04
# ###

import torch
from torch_geometric.nn import Linear, MessagePassing

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
        self.out_size = out_size
        self.is_first = is_first

        self.lin_h = Linear(out_size, att_size, bias=False, weight_initializer='glorot')
        self.lin_x_init = Linear(init_size, att_size, bias=False, weight_initializer='glorot')
        self.lin_key = Linear(f_size, att_size, bias=False, weight_initializer='glorot')
        self.lin_value = Linear(f_size, out_size, bias=False, weight_initializer='glorot')
        self.attention = Attention(dim=1, dropout_p=dropout_p)

    def reset_parameters(self):
        self.lin_h.reset_parameters()
        self.lin_x_init.reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_value.reset_parameters()

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
        h = self.lin_h(h)
        x_init = self.lin_x_init(x_init)
        c, alpha = self.propagate(edge_index, f=f, h=h, x_init=x_init, edge_type=edge_type, x_batch=x_batch, f_batch=f_batch)
        return c, alpha

    def message(self, h_j, x_init_j, edge_index, edge_type):
        """
        Merges each nodes current features with parents and brothers initial embeddings.
        :param h_j: edge source node features after last GCN
        :param x_init_j: edge source node initial embedding
        :param edge_index: edge index
        :param edge_type: edge types list
        :return: query for attention
        """
        pc_edges_mask = (edge_type == int(SltEdgeTypes.PARENT_CHILD)).to(torch.long).unsqueeze(1)
        cc_edges_mask = (edge_type == int(SltEdgeTypes.CURRENT_CURRENT)).to(torch.long).unsqueeze(1)
        bb_edges_mask = (edge_type == int(SltEdgeTypes.LEFTBROTHER_RIGHTBROTHER)).to(torch.long).unsqueeze(1)

        query_h = h_j * cc_edges_mask
        query_bro = x_init_j * bb_edges_mask
        query_pa = x_init_j * pc_edges_mask

        query = query_h + query_bro + query_pa
        return query

    def update(self, query, f, x_batch, f_batch):
        """
        Attends to source graph.
        :param query:
        :param f: source graph node features
        :param x_batch: output graph node batch indices
        :param f_batch: source graph node batch indices
        :return: context vectors, attention coefficients
        """
        key = self.lin_key(f)
        value = self.lin_value(f)

        alpha_not_batch_mask = (x_batch.unsqueeze(1) - f_batch.unsqueeze(0) != 0).long()
        context, attn = self.attention(query, key, value, alpha_not_batch_mask)
        return context, attn

