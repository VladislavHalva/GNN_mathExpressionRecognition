from torch import nn

from src.model.Decoder import Decoder
from src.model.Encoder import Encoder

class Model(nn.Module):
    def __init__(
            self, device, edge_features, edge_h_size,
            enc_in_size, enc_h_size, enc_out_size, dec_in_size, dec_h_size, emb_size, dec_att_size,
            vocab_size, end_node_token_id, tokenizer,
            enc_vgg_dropout_p, enc_gat_dropout_p, dec_emb_dropout_p, dec_att_dropout_p):
        super(Model, self).__init__()

        self.encoder = Encoder(edge_features, edge_h_size, enc_in_size, enc_h_size, enc_out_size, vocab_size, enc_vgg_dropout_p, enc_gat_dropout_p)
        self.decoder = Decoder(device, enc_out_size, dec_in_size, dec_h_size, emb_size, dec_att_size, vocab_size, end_node_token_id, tokenizer, dec_emb_dropout_p, dec_att_dropout_p)

        self.lin_x_out = nn.Linear(enc_out_size, vocab_size, bias=False)

    def forward(self, data, beam_search=True, beam_width=3):
        x, edge_index, edge_attr, x_conv_score = self.encoder(data.x, data.edge_index, data.edge_attr)
        y, y_batch, y_edge_index, y_edge_type, y_score, y_edge_rel_score, gcn1_alpha, gcn2_alpha, gcn3_alpha, = \
            self.decoder(x, data.x_batch, data.tgt_y, data.tgt_edge_index, data.tgt_edge_type, data.tgt_y_batch, beam_search, beam_width)

        data.x = x
        data.x_score = self.lin_x_out(x)
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.x_conv_score = x_conv_score

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

