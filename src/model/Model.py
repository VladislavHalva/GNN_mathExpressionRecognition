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
        data = self.encoder(data)
        data = self.decoder(data)
        return data

