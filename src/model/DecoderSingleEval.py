import networkx as nx
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from src.definitions.SltEdgeTypes import SltEdgeTypes
from src.definitions.SrtEdgeTypes import SrtEdgeTypes
from src.definitions.exceptions.ModelParamsError import ModelParamsError
from src.model.GCNDecLayer import GCNDecLayer


class DecoderSingleEval(nn.Module):
    def __init__(self, device, f_size, h_size, emb_size, vocab_size, end_node_token_id, tokenizer):
        super(DecoderSingleEval, self).__init__()
        self.device = device
        self.f_size = f_size
        self.h_size = h_size
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.end_node_token_id = end_node_token_id
        self.max_output_graph_size = 70
        self.tokenizer = tokenizer

        self.embeds = nn.Embedding(vocab_size, emb_size)
        self.gcn1 = GCNDecLayer(device, f_size, emb_size, h_size, is_first=True)
        self.gcn2 = GCNDecLayer(device, f_size, h_size, h_size, is_first=False)
        self.gcn3 = GCNDecLayer(device, f_size, h_size, emb_size, is_first=False)

        self.lin_z_out = Linear(emb_size, vocab_size, bias=True)
        self.lin_g_out = Linear(2 * emb_size, len(SrtEdgeTypes))

    def forward(self, x, x_batch, tgt_y=None, tgt_edge_index=None, tgt_edge_type=None, tgt_y_batch=None):
        if self.training:
            if \
                    not torch.is_tensor(tgt_y) or not torch.is_tensor(tgt_edge_index) or \
                    not torch.is_tensor(tgt_edge_type) or not torch.is_tensor(tgt_y_batch):
                raise ModelParamsError('ground truth SLT graph missing while training')
            # create tgt nodes embeddings
            y = self.embeds(tgt_y.unsqueeze(1))
            # remove dimension added by embedding layer
            y = y.squeeze(1)
            print(torch.sum(y[0]))
            # copy embeds for loss
            # embeds = None
            embeds = self.lin_z_out(y)
            # rename to be consistent with evaluation time
            y_edge_index = tgt_edge_index
            y_edge_type = tgt_edge_type
            y_batch = tgt_y_batch
            # gcn layers
            y = self.gcn1(x, y, y_edge_index, y_edge_type, x_batch, y_batch)
            gcn1_alpha = self.gcn1.alpha_values
            y = self.gcn2(x, y, y_edge_index, y_edge_type, x_batch, y_batch)
            gcn2_alpha = self.gcn2.alpha_values
            y = self.gcn3(x, y, y_edge_index, y_edge_type, x_batch, y_batch)
            gcn3_alpha = self.gcn3.alpha_values
        else:
            embeds = None
            y, y_edge_index, y_edge_type = self.gen_graph(x)
            gcn1_alpha, gcn2_alpha, gcn3_alpha = None, None, None

        # predictions for nodes from output graph
        y_score = self.lin_z_out(y)
        # build output graph edge features by concatenation corresponding nodes features
        y_edge_features = y[y_edge_index].permute(1, 0, 2)
        y_edge_features = y_edge_features.flatten(1)
        # predictions for edges from output graph
        y_edge_rel_score = self.lin_g_out(y_edge_features)
        return y, y_edge_index, y_edge_type, y_score, y_edge_rel_score, embeds, gcn1_alpha, gcn2_alpha, gcn3_alpha

    def gen_graph(self, x):
        # holds the state of graph nodes as it should look before gcn processing
        y_init = torch.tensor([], dtype=torch.float).to(self.device)
        # holds the state of graph nodes as it should look after gcn processing
        y = torch.tensor([], dtype=torch.float).to(self.device)
        y_eindex = torch.tensor([[], []], dtype=torch.long).to(self.device)
        y_etype = torch.zeros(0, dtype=torch.long).to(self.device)
        y_init, y, y_eindex, y_etype, _, _ = self.gen_subtree(x, y_init, y, y_eindex, y_etype, None, None, None)
        return y, y_eindex, y_etype

    def create_edge(self, y_eindex, y_etype, src_id, tgt_id, etype):
        edge = torch.tensor([[src_id], [tgt_id]], dtype=torch.long).to(self.device)
        edge_type = torch.tensor([etype], dtype=torch.long).to(self.device)
        y_eindex = torch.cat([y_eindex, edge], dim=1)
        y_etype = torch.cat([y_etype, edge_type], dim=0)
        return y_eindex, y_etype

    def gen_subtree(self, x, y_init, y, y_eindex, y_etype, pa_id, gp_id, ls_id):
        # create new node
        i = y.shape[0]
        y_i = torch.zeros((1, self.emb_size), dtype=torch.float).to(self.device)
        y_init = torch.cat([y_init, y_i], dim=0)
        y = torch.cat([y, y_i], dim=0)
        # connect node to graph
        y_eindex, y_etype = self.create_edge(y_eindex, y_etype, i, i, SltEdgeTypes.CURRENT_CURRENT)
        if pa_id is not None:
            y_eindex, y_etype = self.create_edge(y_eindex, y_etype, pa_id, i, SltEdgeTypes.PARENT_CHILD)
        if gp_id is not None:
            y_eindex, y_etype = self.create_edge(y_eindex, y_etype, gp_id, i, SltEdgeTypes.GRANDPARENT_GRANDCHILD)
        if ls_id is not None:
            y_eindex, y_etype = self.create_edge(y_eindex, y_etype, ls_id, i, SltEdgeTypes.LEFTBROTHER_RIGHTBROTHER)
        # lets say that everything belongs to one batch
        x_batch = torch.zeros(x.shape[0], dtype=torch.long).to(self.device)
        y_batch = torch.zeros(y.shape[0], dtype=torch.long).to(self.device)
        # process with GCNs
        y_processed = torch.clone(y_init)
        y_processed = self.gcn1(x, y_processed, y_eindex, y_etype, x_batch, y_batch)
        y_processed = self.gcn2(x, y_processed, y_eindex, y_etype, x_batch, y_batch)
        y_processed = self.gcn3(x, y_processed, y_eindex, y_etype, x_batch, y_batch)
        # update value only for newly created node
        y[i] = y_processed[i]        # decode newly generated node
        y_i_score = self.lin_z_out(y[i].unsqueeze(0))
        y_i_token = y_i_score.squeeze(0).argmax(dim=0)
        y_i_embed = self.embeds(y_i_token.unsqueeze(0).unsqueeze(0))
        y_init[i] = y_i_embed.squeeze(0).squeeze(0)
        if y_i_token == self.end_node_token_id:
            # if node is end leaf stop subtree and return (with True for "most recent is end-leaf")
            return y_init, y, y_eindex, y_etype, i, True
        else:
            subl_ls_id = None
            subl_end = False
            while not subl_end and y.shape[0] <= self.max_output_graph_size:
                # generate sublevel with subtree for each node, until end node is generated
                y_init, y, y_eindex, y_etype, subl_ls_id, subl_end = self.gen_subtree(x, y_init, y, y_eindex, y_etype, i, pa_id, subl_ls_id)
            return y_init, y, y_eindex, y_etype, i, False
