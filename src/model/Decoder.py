from itertools import compress
import torch
from torch import nn
from torch.nn import Linear

from src.definitions.SltEdgeTypes import SltEdgeTypes
from src.definitions.SrtEdgeTypes import SrtEdgeTypes
from src.definitions.exceptions.ModelParamsError import ModelParamsError
from src.model.GCNDecLayer import GCNDecLayer


class Decoder(nn.Module):
    def __init__(self, device, f_size, h_size, emb_size, vocab_size, end_node_token_id, tokenizer):
        super(Decoder, self).__init__()
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
            y, y_batch, y_edge_index, y_edge_type = self.gen_graph(x, x_batch)
            gcn1_alpha, gcn2_alpha, gcn3_alpha = None, None, None

        # predictions for nodes from output graph
        y_score = self.lin_z_out(y)
        # build output graph edge features by concatenation corresponding nodes features
        y_edge_features = y[y_edge_index].permute(1, 0, 2)
        y_edge_features = y_edge_features.flatten(1)
        # predictions for edges from output graph
        y_edge_rel_score = self.lin_g_out(y_edge_features)
        return y, y_batch, y_edge_index, y_edge_type, y_score, y_edge_rel_score, gcn1_alpha, gcn2_alpha, gcn3_alpha

    def gen_graph(self, x, x_batch):
        # get batch size to generate all SLT trees at once
        bs = torch.unique(x_batch).shape[0]
        # holds the state of graph nodes as it should look before gcn processing
        y_init = torch.tensor([], dtype=torch.float).to(self.device)
        # holds the state of graph nodes as it should look after gcn processing
        y = torch.tensor([], dtype=torch.float).to(self.device)
        y_batch = torch.tensor([], dtype=torch.long).to(self.device)
        y_eindex = torch.tensor([[], []], dtype=torch.long).to(self.device)
        y_etype = torch.zeros(0, dtype=torch.long).to(self.device)
        pa_ids, gp_ids, ls_ids = [None] * bs, [None] * bs, [None] * bs
        gen_tree = [True] * bs
        y_init, y, y_batch, y_eindex, y_etype, _, _ = self.gen_subtree(x, x_batch, y_init, y, y_batch, y_eindex, y_etype, gen_tree, pa_ids, gp_ids, ls_ids)
        return y, y_batch, y_eindex, y_etype

    def create_edge(self, y_eindex, y_etype, src_id, tgt_id, etype):
        edge = torch.tensor([[src_id], [tgt_id]], dtype=torch.long).to(self.device)
        edge_type = torch.tensor([etype], dtype=torch.long).to(self.device)
        y_eindex = torch.cat([y_eindex, edge], dim=1)
        y_etype = torch.cat([y_etype, edge_type], dim=0)
        return y_eindex, y_etype

    def gen_subtree(self, x, x_batch, y_init, y, y_batch, y_eindex, y_etype, gen_tree, pa_ids, gp_ids, ls_ids):
        # create new nodes for each batch that is not finished yet
        batch_idx_unfinisned = torch.tensor(list(compress(range(len(gen_tree)), gen_tree)), dtype=torch.long).to(self.device)
        y_new = torch.zeros((len(batch_idx_unfinisned), self.emb_size), dtype=torch.float).to(self.device)
        y_new_idx = torch.tensor([y.shape[0] + order for order in range(y_new.shape[0])], dtype=torch.long).to(self.device)
        y_init = torch.cat([y_init, y_new], dim=0)
        y = torch.cat([y, y_new], dim=0)
        y_batch = torch.cat([y_batch, batch_idx_unfinisned], dim=0)

        # connect nodes to graph
        for i, i_batch in enumerate(batch_idx_unfinisned):
            y_eindex, y_etype = self.create_edge(y_eindex, y_etype, y_new_idx[i], y_new_idx[i], SltEdgeTypes.CURRENT_CURRENT)
            if pa_ids[i_batch] is not None:
                y_eindex, y_etype = self.create_edge(y_eindex, y_etype, pa_ids[i_batch], y_new_idx[i], SltEdgeTypes.PARENT_CHILD)
            if gp_ids[i_batch] is not None:
                y_eindex, y_etype = self.create_edge(y_eindex, y_etype, gp_ids[i_batch], y_new_idx[i], SltEdgeTypes.GRANDPARENT_GRANDCHILD)
            if ls_ids[i_batch] is not None:
                y_eindex, y_etype = self.create_edge(y_eindex, y_etype, ls_ids[i_batch], y_new_idx[i], SltEdgeTypes.LEFTBROTHER_RIGHTBROTHER)
        # process with GCNs
        y_processed = torch.clone(y_init)
        y_processed = self.gcn1(x, y_processed, y_eindex, y_etype, x_batch, y_batch)
        y_processed = self.gcn2(x, y_processed, y_eindex, y_etype, x_batch, y_batch)
        y_processed = self.gcn3(x, y_processed, y_eindex, y_etype, x_batch, y_batch)
        # update value only for newly created node
        y[y_new_idx] = y_processed[y_new_idx]
        # decode newly generated node
        y_new_score = self.lin_z_out(y[y_new_idx])
        y_new_token = y_new_score.argmax(dim=1)
        y_new_embed = self.embeds(y_new_token.unsqueeze(1))
        y_init[y_new_idx] = y_new_embed.squeeze(1)
        y_new_idx_per_batch = [None] * len(pa_ids)
        this_leaf = [False] * len(pa_ids)
        # determine for which batch items the generating process should continue
        # and create list of current node ids per batch item
        gen_subtree = list(gen_tree)
        for i, i_batch in enumerate(batch_idx_unfinisned):
            if y_new_token[i] == self.end_node_token_id:
                gen_subtree[i_batch] = False
                this_leaf[i_batch] = True
            y_new_idx_per_batch[i_batch] = y_new_idx[i]
        if not any(gen_subtree):
            # end leaf node generated for all batch items
            return y_init, y, y_batch, y_eindex, y_etype, y_new_idx_per_batch, this_leaf
        else:
            subl_ls_ids = [None] * len(ls_ids)
            while any(gen_subtree):
                # generate sublevel with subtree for each node, until end node is generated
                y_init, y, y_batch, y_eindex, y_etype, subl_ls_ids, last_leaf = \
                    self.gen_subtree(x, x_batch, y_init, y, y_batch, y_eindex, y_etype, gen_subtree, y_new_idx_per_batch, pa_ids, subl_ls_ids)
                # stop batch items tree generation for those batch items, whose trees reached maximal nodes count
                batch_items_nodes_counts = torch.unique(y_batch, return_counts=True, sorted=True)[1]
                batch_items_nodes_limit_reached = (batch_items_nodes_counts > self.max_output_graph_size)
                # update for which batch items the subtree generation shall continue
                gen_subtree = [True if gen_subtree_i and not last_leaf[i] and not batch_items_nodes_limit_reached[i] else False for i, gen_subtree_i in enumerate(gen_subtree)]
            return y_init, y, y_batch, y_eindex, y_etype, y_new_idx_per_batch, this_leaf
