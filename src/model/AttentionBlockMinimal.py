# ###
# Mathematical expression recognition tool.
# Written as a part of masters thesis at VUT FIT Brno, 2022

# Author: Vladislav Halva
# Login: xhalva04
# ###

from torch import nn
from torch_geometric.nn import Linear

from src.model.Attention import Attention


class AttentionBlockMinimal(nn.Module):
    """
    Decoder source graph attention block, which computes query only based on current nodes features.
    """
    def __init__(self, device, f_size, out_size, init_size, dropout_p, is_first=False):
        super(AttentionBlockMinimal, self).__init__()
        self.device = device
        self.dropout_p = dropout_p
        self.negative_slope = 0.2
        self.out_size = out_size
        self.is_first = is_first

        self.lin_h = Linear(out_size, 128, bias=False, weight_initializer='glorot')
        self.lin_key = Linear(f_size, 128, bias=False, weight_initializer='glorot')
        self.lin_value = Linear(f_size, out_size, bias=False, weight_initializer='glorot')
        self.attention = Attention(dim=2, dropout_p=dropout_p)

    def reset_parameters(self):
        self.lin_h.reset_parameters()
        self.lin_key.reset_parameters()
        self.lin_value.reset_parameters()

    def forward(self, f, x, h, edge_index, edge_type, f_batch, x_batch, x_init):
        h = self.lin_h(h)
        query = h
        key = self.lin_key(f)
        value = self.lin_value(f)

        alpha_not_batch_mask = (x_batch.unsqueeze(1) - f_batch.unsqueeze(0) != 0).long()
        context, attn = self.attention(query, key, value, alpha_not_batch_mask)
        return context, attn




