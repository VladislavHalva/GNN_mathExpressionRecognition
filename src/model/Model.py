from torch import nn

from src.model.Decoder import Decoder
from src.model.Encoder import Encoder


class Model(nn.Module):
    def __init__(self, components_shape, input_edge_size, input_feature_size, hidden_size, embed_size, vocab_size, end_node_token_id):
        super(Model, self).__init__()

        self.encoder = Encoder(components_shape, input_edge_size, input_feature_size, hidden_size, embed_size)
        self.decoder = Decoder(hidden_size, embed_size, vocab_size, end_node_token_id)

    def forward(self, data):
        data = self.encoder(data)
        data = self.decoder(data)
        return data

