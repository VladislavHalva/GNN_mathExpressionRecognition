import torch
from torch import nn

from src.model.Decoder import Decoder
from src.model.Encoder import Encoder


class Model(nn.Module):
    def __init__(
            self, device, components_shape, edge_features,
            enc_in_size, enc_h_size, enc_out_size, dec_h_size, emb_size,
            vocab_size, end_node_token_id):
        super(Model, self).__init__()

        self.encoder = Encoder(device, components_shape, edge_features, enc_in_size, enc_h_size, enc_out_size)
        self.decoder = Decoder(device, enc_out_size, dec_h_size, emb_size, vocab_size, end_node_token_id)

    def forward(self, data):
        x, edge_index, edge_attr = self.encoder(data.x, data.edge_index, data.edge_attr)
        y, y_edge_index, y_edge_type, y_score, y_edge_rel_score, embeds = self.decoder(x, data.x_batch, data.tgt_y, data.tgt_edge_index, data.tgt_edge_type, data.tgt_y_batch)

        data.x = x
        data.edge_index = edge_index
        data.edge_attr = edge_attr

        data.y = y
        data.y_edge_index = y_edge_index
        data.y_edge_type = y_edge_type
        data.y_score = y_score
        data.y_edge_rel_score = y_edge_rel_score
        data.embeds = embeds

        return data

