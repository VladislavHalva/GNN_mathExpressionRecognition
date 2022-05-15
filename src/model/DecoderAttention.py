import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class DecoderAttention(nn.Module):
    def __init__(self, dim, dropout_p):
        super(DecoderAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)
        self.dropout_p = dropout_p

    def forward(self, query, key, value, mask=None):
        score = torch.mm(query, key.transpose(0, 1)) / self.sqrt_dim

        if mask is not None:
            score.masked_fill_(mask.view(score.size()).bool(), -float('Inf'))

        attn = F.softmax(score, -1)
        attn = F.dropout(attn, p=self.dropout_p, training=self.training)
        context = torch.mm(attn, value)
        return context, attn
