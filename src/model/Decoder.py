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


class Decoder(nn.Module):
    def __init__(self, device, f_size, h_size, emb_size, vocab_size, end_node_token_id):
        super(Decoder, self).__init__()
        self.device = device
        self.f_size = f_size
        self.h_size = h_size
        self.emb_size = emb_size
        self.vocab_size = vocab_size
        self.end_node_token_id = end_node_token_id
        self.max_output_graph_size = 70

        self.embeds = nn.Embedding(vocab_size, emb_size)
        self.gcn1 = GCNDecLayer(device, f_size, emb_size, h_size, is_first=True)
        self.gcn2 = GCNDecLayer(device, f_size, h_size, h_size, is_first=False)
        self.gcn3 = GCNDecLayer(device, f_size, h_size, emb_size, is_first=False)

        self.lin_z_out = Linear(emb_size, vocab_size, bias=True)
        self.lin_g_out = Linear(2 * emb_size, len(SrtEdgeTypes))

    def forward(self, x, x_batch, tgt_y=None, tgt_edge_index=None, tgt_edge_type=None, tgt_y_batch=None):
        if self.training or True:
            if \
                    not torch.is_tensor(tgt_y) or not torch.is_tensor(tgt_edge_index) or \
                    not torch.is_tensor(tgt_edge_type) or not torch.is_tensor(tgt_y_batch):
                raise ModelParamsError('ground truth SLT graph missing while training')
            # create tgt nodes embeddings
            y = self.embeds(tgt_y)
            # sums token embeddings to one in case of multiple tokens per node
            y = y.squeeze(1)
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
            y, y_edge_index, y_edge_type = self.generate_output_graph(x)
            gcn1_alpha, gcn2_alpha, gcn3_alpha = None, None, None

        # predictions for nodes from output graph
        y_score = self.lin_z_out(y)
        # build output graph edge features by concatenation corresponding nodes features
        y_edge_features = y[y_edge_index].permute(1, 0, 2)
        y_edge_features = y_edge_features.flatten(1)
        # predictions for edges from output graph
        y_edge_rel_score = self.lin_g_out(y_edge_features)
        return y, y_edge_index, y_edge_type, y_score, y_edge_rel_score, embeds, gcn1_alpha, gcn2_alpha, gcn3_alpha

    def generate_output_graph(self, x):
        y = torch.zeros(0, dtype=torch.float).to(self.device)
        y_eindex = torch.tensor([[], []], dtype=torch.long).to(self.device)
        y_etype = torch.zeros(0, dtype=torch.long).to(self.device)
        y, y_eindex, y_etype, _, _ = self.generate_output_subtree(x, y, y_eindex, y_etype, None, None, None)
        return y, y_eindex, y_etype

    def append_edge_with_type(self, e_index, e_type, src_id, tgt_id, etype):
        new_edge = torch.tensor([[src_id], [tgt_id]], dtype=torch.long).to(self.device)
        new_etype = torch.tensor([etype], dtype=torch.long).to(self.device)
        e_index = torch.cat([e_index, new_edge], dim=1)
        e_type = torch.cat([e_type, new_etype], dim=0)
        return e_index, e_type

    def generate_output_subtree(self, x, y, y_eindex, y_etype, pa_id, gp_id, ls_id):
        # generate new node and add it to graph nodes

        nodes_count = y.size(0)
        y_new_id = nodes_count
        y_new = torch.zeros((1, self.emb_size), dtype=torch.float).to(self.device)
        y = torch.cat([y, y_new], dim=0)

        # connect new node to existing graph with edges
        y_eindex, y_etype = self.append_edge_with_type(y_eindex, y_etype, y_new_id, y_new_id,
                                                       SltEdgeTypes.CURRENT_CURRENT)
        if pa_id is not None:
            y_eindex, y_etype = self.append_edge_with_type(y_eindex, y_etype, pa_id, y_new_id,
                                                           SltEdgeTypes.PARENT_CHILD)
        if gp_id is not None:
            y_eindex, y_etype = self.append_edge_with_type(y_eindex, y_etype, gp_id, y_new_id,
                                                           SltEdgeTypes.GRANDPARENT_GRANDCHILD)
        if ls_id is not None:
            y_eindex, y_etype = self.append_edge_with_type(y_eindex, y_etype, ls_id, y_new_id,
                                                           SltEdgeTypes.LEFTBROTHER_RIGHTBROTHER)

        # run through graph convolutional layers
        x_batch = torch.zeros(x.size(0), dtype=torch.long).to(self.device)
        y_batch = torch.zeros(y.size(0), dtype=torch.long).to(self.device)
        y_modified = torch.clone(y)
        y_modified = self.gcn1(x, y_modified, y_eindex, y_etype, x_batch, y_batch)
        y_modified = self.gcn2(x, y_modified, y_eindex, y_etype, x_batch, y_batch)
        y_modified = self.gcn3(x, y_modified, y_eindex, y_etype, x_batch, y_batch)

        # update feature for the newly generated node
        # create update and preserve masks
        update_mask = torch.zeros(y.size(0), dtype=torch.long).to(self.device)
        preserve_mask = torch.ones(y.size(0), dtype=torch.long).to(self.device)
        update_mask[y_new_id] = 1
        preserve_mask[y_new_id] = 0
        update_mask = update_mask.unsqueeze(1)
        preserve_mask = preserve_mask.unsqueeze(1)
        # mask features in either old or new values and merge them (sum - each feature vector is zeros in one of the tensors)
        y = y * preserve_mask
        y_modified = y_modified * update_mask
        y = y + y_modified

        # decode new node token and check if end leaf node
        y_new = y[y_new_id]
        y_new = self.lin_z_out(y_new.unsqueeze(0))
        y_new = F.softmax(y_new, dim=1)
        y_new = y_new.squeeze(0)
        _, predicted_token = y_new.max(dim=0)
        predicted_token = predicted_token.item()
        is_leaf = predicted_token == self.end_node_token_id
        nodes_count = y.size(0)

        if not is_leaf and nodes_count < self.max_output_graph_size:
            # if generated node is not end leaf node - generate subtree
            ls_id = None
            sublevel_leaf_generated = False
            while not sublevel_leaf_generated and nodes_count < self.max_output_graph_size:
                # keep generating sub-level nodes (and their subtrees) until direct child end leaf node is generated
                y, y_eindex, y_etype, ls_id, sublevel_leaf_generated = \
                    self.generate_output_subtree(x, y, y_eindex, y_etype, y_new_id, pa_id, ls_id)
                nodes_count = y.size(0)

        return y, y_eindex, y_etype, y_new_id, is_leaf
