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
from src.model.GCNDecLayer import GCNDecLayer


class Decoder(nn.Module):
    def __init__(self, device, hidden_size, embed_size, vocab_size, end_node_token_id):
        super(Decoder, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.end_node_token_id = end_node_token_id
        self.max_output_graph_size = 70

        self.embeds = nn.Embedding(vocab_size, embed_size)
        self.gcn1 = GCNDecLayer(device, hidden_size, embed_size, is_first=True)
        self.gcn2 = GCNDecLayer(device, hidden_size, embed_size, is_first=False)

        self.lin_z_out = Linear(embed_size, vocab_size, bias=True)
        self.lin_g_out = Linear(2 * embed_size, len(SrtEdgeTypes))

    def forward(self, data):
        if self.training:
            # create tgt nodes embeddings
            data.out_x = self.embeds(data.tgt_x)
            # sums token embeddings to one in case of multiple tokens per node
            data.out_x = torch.sum(data.out_x, dim=1)
            # gcn layers
            data.out_x = self.gcn1(data.x, data.out_x, data.tgt_edge_index, data.tgt_edge_type)
            data.out_x = self.gcn2(data.x, data.out_x, data.tgt_edge_index, data.tgt_edge_type)
        else:
            data.out_x, data.tgt_edge_index, data.tgt_edge_type = self.generate_output_graph(data)

        # predictions for nodes from output graph
        out_node_predictions = self.lin_z_out(data.out_x)
        out_node_predictions = F.log_softmax(out_node_predictions, dim=1)
        data.out_x_pred = out_node_predictions
        # predictions for edges from output graph
        out_edge_features = data.out_x[data.tgt_edge_index].permute(1, 0, 2)
        out_edge_features = out_edge_features.flatten(1)
        out_edge_predictions = self.lin_g_out(out_edge_features)
        out_edge_predictions = F.log_softmax(out_edge_predictions, dim=1)
        data.out_edge_pred = out_edge_predictions
        return data

    def generate_output_graph(self, data):
        out_x = torch.zeros(0, dtype=torch.float).to(self.device)
        tgt_edge_index = torch.zeros(0, dtype=torch.long).to(self.device)
        tgt_edge_type = torch.zeros(0, dtype=torch.long).to(self.device)
        out_x, tgt_edge_index, tgt_edge_type, _, _ = self.generate_output_subtree(data, out_x, tgt_edge_index,
                                                                                  tgt_edge_type, None, None, None)
        return out_x, tgt_edge_index, tgt_edge_type

    def generate_output_subtree(self, data, x, edge_index, edge_type, x_root_id, x_root_parent_id,
                                x_root_last_child_id):
        # DFS traversal graph generation
        # generate new node and add it to graph nodes
        new_node_id = x.size(0)
        new_node = torch.zeros((1, self.embed_size), dtype=torch.float).to(self.device)
        x = torch.cat([x, new_node], dim=0)

        # add self-loop edge
        cc_edge = torch.tensor([[new_node_id], [new_node_id]], dtype=torch.long).to(self.device)
        edge_index = torch.cat([edge_index, cc_edge], dim=1)
        edge_type = torch.cat([edge_type, torch.tensor([SltEdgeTypes.CURRENT_CURRENT], dtype=torch.long).to(self.device)])
        # add parent-child edge
        if x_root_id:
            pc_edge = torch.tensor([[x_root_id], [new_node_id]], dtype=torch.long).to(self.device)
            edge_index = torch.cat([edge_index, pc_edge], dim=1)
            edge_type = torch.cat([edge_type, torch.tensor([SltEdgeTypes.PARENT_CHILD], dtype=torch.long).to(self.device)])
        # add grandparent-grandchild edge
        if x_root_parent_id:
            gg_edge = torch.tensor([[x_root_parent_id], [new_node_id]], dtype=torch.long).to(self.device)
            edge_index = torch.cat([edge_index, gg_edge], dim=1)
            edge_type = torch.cat([edge_type, torch.tensor([SltEdgeTypes.GRANDPARENT_GRANDCHILD], dtype=torch.long).to(self.device)])
        # add leftBrother-rightBrother edge
        if x_root_last_child_id:
            bb_edge = torch.tensor([[x_root_last_child_id], [new_node_id]], dtype=torch.long).to(self.device)
            edge_index = torch.cat([edge_index, bb_edge], dim=1)
            edge_type = torch.cat([edge_type, torch.tensor([SltEdgeTypes.LEFTBROTHER_RIGHTBROTHER], dtype=torch.long).to(self.device)])

        # run through graph convolutional layers
        x = self.gcn1(data.x, x, edge_index, edge_type, torch.tensor([new_node_id], dtype=torch.long))
        x = self.gcn2(data.x, x, edge_index, edge_type, torch.tensor([new_node_id], dtype=torch.long))

        # decode new node token and check if end leaf node
        new_node_features = x[new_node_id].unsqueeze(0)
        new_node_features = self.lin_z_out(new_node_features)
        new_node_features = F.softmax(new_node_features, dim=1)
        _, max_id = new_node_features.max(dim=1)
        max_id = max_id.item()

        if max_id == self.end_node_token_id or x.size(0) == self.max_output_graph_size:
            return x, edge_index, edge_type, True, None
        else:
            last_child_id = None
            while x.size(0) < self.max_output_graph_size:
                x, edge_index, edge_type, is_leaf, last_child_id = self.generate_output_subtree(data, x, edge_index,
                                                                                                edge_type, new_node_id,
                                                                                                x_root_id,
                                                                                                last_child_id)
                if is_leaf:
                    break
            return x, edge_index, edge_type, False, new_node_id
