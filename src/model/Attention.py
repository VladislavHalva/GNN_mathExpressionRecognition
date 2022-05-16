# ###
# Mathematical expression recognition tool.
# Written as a part of masters thesis at VUT FIT Brno, 2022

# Author: Vladislav Halva
# Login: xhalva04
# ###

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Scaled dot product attention module.
    """
    def __init__(self, dim, dropout_p):
        super(Attention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)
        self.dropout_p = dropout_p

    def forward(self, query, key, value, mask=None):
        """
        :param query: attention query
        :param key: attention key
        :param value: attention value
        :param mask: mask - where mask is zero attention will be masked
        :return:
        """
        score = torch.mm(query, key.transpose(0, 1)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()).bool(), -float('Inf'))

        attn = F.softmax(score, -1)
        attn = F.dropout(attn, p=self.dropout_p, training=self.training)
        context = torch.mm(attn, value)
        return context, attn
