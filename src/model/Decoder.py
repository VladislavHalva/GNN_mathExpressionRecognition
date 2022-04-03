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
        if self.training:
            if \
                    not torch.is_tensor(tgt_y) or not torch.is_tensor(tgt_edge_index) or \
                    not torch.is_tensor(tgt_edge_type) or not torch.is_tensor(tgt_y_batch):
                raise ModelParamsError('ground truth SLT graph missing while training')
            # create tgt nodes embeddings
            y = self.embeds(tgt_y)
            # sums token embeddings to one in case of multiple tokens per node
            y = torch.sum(y, dim=1)
            # rename to be consistent with evaluation time
            y_edge_index = tgt_edge_index
            y_edge_type = tgt_edge_type
            y_batch = tgt_y_batch
            # gcn layers
            y = self.gcn1(x, y, y_edge_index, y_edge_type, x_batch, y_batch)
            y = self.gcn2(x, y, y_edge_index, y_edge_type, x_batch, y_batch)
            y = self.gcn3(x, y, y_edge_index, y_edge_type, x_batch, y_batch)
        else:
            y, y_edge_index, y_edge_type, y_batch = self.generate_output_graph(x, x_batch)

        # predictions for nodes from output graph
        y_pred = self.lin_z_out(y)
        y_pred = F.log_softmax(y_pred, dim=1)
        # build output graph edge features by concatenation corresponding nodes features
        y_edge_features = y[y_edge_index].permute(1, 0, 2)
        y_edge_features = y_edge_features.flatten(1)
        # predictions for edges from output graph
        y_edge_pred = self.lin_g_out(y_edge_features)
        y_edge_pred = F.log_softmax(y_edge_pred, dim=1)
        return y, y_edge_index, y_edge_type, y_pred, y_edge_pred

    def generate_output_graph(self, x, x_batch):
        y = torch.zeros(0, dtype=torch.float).to(self.device)
        y_eindex = torch.zeros(0, dtype=torch.long).to(self.device)
        y_etype = torch.zeros(0, dtype=torch.long).to(self.device)
        y, y_eindex, y_etype, _, _ = self.generate_output_subtree(x, y, y_eindex,
                                                                  y_etype, None, None, None)

        return y, y_eindex, y_etype

    def append_edge_with_type(self, edge_index, edge_type, src_id, tgt_id, etype):
        edge = torch.tensor([[src_id], [tgt_id]], dtype=torch.long).to(self.device)
        etype = torch.tensor([etype], dtype=torch.long).to(self.device)
        edge_index = torch.cat([edge_index, edge], dim=1)
        edge_type = torch.cat([edge_type, etype], dim=0)
        return edge_index, edge_type

    def generate_output_subtree(self, x, y, y_eindex, y_etype, parent_id, grandparent_id,
                                last_sibling_id):
        # DFS traversal graph generation

        # generate new node and add it to graph nodes
        y_new_id = y.size(0)
        y_new_id_tensor = torch.tensor([y_new_id], dtype=torch.long).to(self.device)
        y_new = torch.zeros((1, self.emb_size), dtype=torch.float).to(self.device)
        y = torch.cat([y, y_new], dim=0)

        # add self-loop edge
        y_eindex, y_etype = self.append_edge_with_type(y_eindex, y_etype, y_new_id, y_new_id,
                                                       SltEdgeTypes.CURRENT_CURRENT)

        if parent_id:
            # add parent-child edge
            y_eindex, y_etype = self.append_edge_with_type(y_eindex, y_etype, parent_id, y_new_id,
                                                           SltEdgeTypes.PARENT_CHILD)
        if grandparent_id:
            # add grandparent-grandchild edge
            y_eindex, y_etype = self.append_edge_with_type(y_eindex, y_etype, grandparent_id, y_new_id,
                                                           SltEdgeTypes.GRANDPARENT_GRANDCHILD)
        if last_sibling_id:
            # add leftBrother-rightBrother edge
            y_eindex, y_etype = self.append_edge_with_type(y_eindex, y_etype, last_sibling_id, y_new_id,
                                                           SltEdgeTypes.LEFTBROTHER_RIGHTBROTHER)

        # run through graph convolutional layers
        y = self.gcn1(x, y, y_eindex, y_etype, y_new_id_tensor)
        y = self.gcn2(x, y, y_eindex, y_etype, y_new_id_tensor)
        y = self.gcn3(x, y, y_eindex, y_etype, y_new_id_tensor)

        # decode new node token and check if end leaf node
        y_new_features = y[y_new_id].unsqueeze(0)
        y_new_features = self.lin_z_out(y_new_features)
        y_new_features = F.softmax(y_new_features, dim=1)
        _, predicted_token = y_new_features.max(dim=1)
        predicted_token = predicted_token.item()

        if predicted_token == self.end_node_token_id or y.size(0) == self.max_output_graph_size:
            # generated node is leaf or max number of nodes reach --> end traversal
            return y, y_eindex, y_etype, True, None, None
        else:
            # generated node is not leaf --> proceed with subtree/child-nodes generation
            last_child_id = None
            while y.size(0) < self.max_output_graph_size:
                # limit to set max number of nodes, but otherwise keep generating until leaf end node is generated
                y, y_eindex, y_etype, is_leaf, last_child_id = self.generate_output_subtree(x, y, y_eindex,
                                                                                                        y_etype,
                                                                                                        y_new_id,
                                                                                                        parent_id,
                                                                                                        last_child_id)

                if is_leaf:
                    # if leaf end node generated --> stop with this subtree and return to parent
                    break
            return y, y_eindex, y_etype, False, y_new_id, None
