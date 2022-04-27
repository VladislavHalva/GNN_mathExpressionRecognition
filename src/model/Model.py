import torch
from torch import nn
from torch.nn import Linear

from src.model.Encoder import Encoder
from src.model.Decoder import Decoder


class Model(nn.Module):
    def __init__(
            self, device, components_shape, edge_features,
            enc_in_size, enc_h_size, enc_out_size, dec_h_size, emb_size,
            vocab_size, end_node_token_id, tokenizer):
        super(Model, self).__init__()

        self.encoder = Encoder(device, components_shape, edge_features, enc_in_size, enc_h_size, enc_out_size)
        self.decoder = Decoder(device, enc_out_size, dec_h_size, emb_size, vocab_size, end_node_token_id, tokenizer)

        self.lin_x_out = Linear(enc_out_size, vocab_size, bias=True)

    def forward(self, data):
        x, edge_index, edge_attr = self.encoder(data.x, data.edge_index, data.edge_attr)
        y, y_batch, y_edge_index, y_edge_type, y_score, y_edge_rel_score, gcn1_alpha, gcn2_alpha, gcn3_alpha = \
            self.decoder(x, data.x_batch, data.tgt_y, data.tgt_edge_index, data.tgt_edge_type, data.tgt_y_batch)

        data.x = x
        data.x_score = self.lin_x_out(x)
        data.edge_index = edge_index
        data.edge_attr = edge_attr

        data.y = y
        data.y_batch = y_batch
        data.y_edge_index = y_edge_index
        data.y_edge_type = y_edge_type
        data.y_score = y_score
        data.y_edge_rel_score = y_edge_rel_score
        data.gcn1_alpha = gcn1_alpha
        data.gcn2_alpha = gcn2_alpha
        data.gcn3_alpha = gcn3_alpha

        return data

