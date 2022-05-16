# ###
# Mathematical expression recognition tool.
# Written as a part of masters thesis at VUT FIT Brno, 2022

# Author: Vladislav Halva
# Login: xhalva04
# ###

from torch import nn

from src.model.Decoder import Decoder
from src.model.Encoder import Encoder

class Model(nn.Module):
    """
    Encoder-decoder model for mathematical expressions recognition wrapper.
    """
    def __init__(
            self, device, edge_features, edge_h_size,
            enc_in_size, enc_h_size, enc_out_size, dec_in_size, dec_h_size, emb_size, dec_att_size,
            vocab_size, end_node_token_id, tokenizer,
            enc_vgg_dropout_p, enc_gat_dropout_p, dec_emb_dropout_p, dec_att_dropout_p):
        """
        :param device: device
        :param edge_features: in edge features size
        :param edge_h_size: hidden and out edge features size
        :param enc_in_size: encoder GATs in nodes features size
        :param enc_h_size: encoder GATs hidden nodes features size
        :param enc_out_size: encoder GATs out nodes features size
        :param dec_in_size: decoder in node features size
        :param dec_h_size: decoder hidden node features size
        :param emb_size: decoder out node features=embeddings size
        :param dec_att_size: decoder source graph attention vector size
        :param vocab_size: vocabulary size
        :param end_node_token_id: id of end-leaf-node token [EOS]
        :param tokenizer: tokenizer
        :param enc_vgg_dropout_p: dropout probability for encoder VGG linear layers
        :param enc_gat_dropout_p: dropout probability for encoder GAT layers
        :param dec_emb_dropout_p: dropout probability for decoder initial embedding layer
        :param dec_att_dropout_p: dropout probability for decoder attention to source graph
        """
        super(Model, self).__init__()

        self.encoder = Encoder(
            edge_features, edge_h_size,
            enc_in_size, enc_h_size, enc_out_size, vocab_size,
            enc_vgg_dropout_p, enc_gat_dropout_p)
        self.decoder = Decoder(
            device, enc_out_size, dec_in_size, dec_h_size, emb_size,
            dec_att_size, vocab_size, end_node_token_id,
            tokenizer, dec_emb_dropout_p, dec_att_dropout_p)

        self.lin_x_out = nn.Linear(enc_out_size, vocab_size, bias=False)

    def forward(self, data, beam_search=True, beam_width=3):
        """
        :param data: databatch object
        :param beam_search: True if beam search, False otherwise
        :param beam_width: if beam search sets the beam width - number of simultaneously generated output graphs
        :return: output data
        """
        # encode source graph
        x, edge_index, edge_attr, x_conv_score = self.encoder(data.x, data.edge_index, data.edge_attr)
        # decode output graph
        y, y_batch, y_edge_index, y_edge_type, y_score, y_edge_rel_score, gcn1_alpha, gcn2_alpha, gcn3_alpha, = self.decoder(
            x, data.x_batch, data.tgt_y, data.tgt_edge_index, data.tgt_edge_type, data.tgt_y_batch, beam_search, beam_width)
        # get source graph node predictions and update values in databatch object for source graph
        data.x = x
        data.x_score = self.lin_x_out(x)
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.x_conv_score = x_conv_score
        # include output graph in output databatch object
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

