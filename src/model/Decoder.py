# ###
# Mathematical expression recognition tool.
# Written as a part of masters thesis at VUT FIT Brno, 2022

# Author: Vladislav Halva
# Login: xhalva04
# ###

import operator
from itertools import compress
from operator import itemgetter

import torch
from torch import nn
import torch.nn.functional as F

from src.data.SltParser import SltParser
from src.definitions.SltEdgeTypes import SltEdgeTypes
from src.definitions.SrtEdgeTypes import SrtEdgeTypes
from src.definitions.exceptions.ModelParamsError import ModelParamsError
from src.model.DecoderBlock import DecoderBlock


class Decoder(nn.Module):
    def __init__(self,
                 device,
                 f_size,
                 in_size,
                 h_size,
                 emb_size,
                 att_size,
                 vocab_size,
                 end_node_token_id,
                 tokenizer,
                 emb_dropout_p,
                 att_dropout_p):
        super(Decoder, self).__init__()
        # settings
        self.device = device
        self.f_size = f_size
        self.in_size = in_size
        self.h_size = h_size
        self.emb_size = emb_size
        self.att_size = att_size
        self.vocab_size = vocab_size
        self.end_node_token_id = end_node_token_id
        self.max_output_graph_size = 70
        self.tokenizer = tokenizer
        self.emb_dropout_p = emb_dropout_p
        self.beam_search = True
        self.beam_width = 3
        # embedding layer and decoder blocks
        self.embeds = nn.Embedding(vocab_size, in_size)
        self.decBlock1 = DecoderBlock(device, f_size, in_size, h_size, att_size, att_dropout_p, in_size, is_first=True)
        self.decBlock2 = DecoderBlock(device, f_size, h_size, h_size, att_size, att_dropout_p, in_size, is_first=False)
        self.decBlock3 = DecoderBlock(device, f_size, h_size, emb_size, att_size, att_dropout_p, in_size, is_first=False)
        # output linear layers for node and edge predictions
        self.lin_node_out = nn.Linear(emb_size, vocab_size, bias=False)
        self.lin_edge_out = nn.Linear(2 * emb_size, len(SrtEdgeTypes), bias=False)
        # properties used to store and return attention coefficients in evaluation time
        self.gcn1_alpha_eval = None
        self.gcn2_alpha_eval = None
        self.gcn3_alpha_eval = None

    def forward(self,
                x,
                x_batch,
                tgt_y=None,
                tgt_edge_index=None,
                tgt_edge_type=None,
                tgt_y_batch=None,
                beam_search=True,
                beam_width=3):
        if self.training:
            if not torch.is_tensor(tgt_y) or not torch.is_tensor(tgt_edge_index) or \
                    not torch.is_tensor(tgt_edge_type) or not torch.is_tensor(tgt_y_batch):
                raise ModelParamsError('ground truth SLT graph missing while training')
            # training time parallel forward pass
            y, y_edge_index, gcn1_alpha, gcn2_alpha, gcn3_alpha = \
                self.train_forward(x, x_batch, tgt_y, tgt_edge_index, tgt_edge_type, tgt_y_batch)
            # renaming
            y_edge_type = tgt_edge_type
            y_batch = tgt_y_batch
            # get predictions
            y_score, y_edge_rel_score = self.get_predictions(y, y_edge_index)
        else:
            self.beam_search = beam_search
            self.beam_width = beam_width
            if beam_search:
                batch_size = torch.unique(x_batch).shape[0]
                if batch_size > 1:
                    raise ModelParamsError('Beam Search does not support mini-batches during evaluation')

                y, y_batch, y_beam, y_edge_index, y_edge_type = self.gen_graph_beam(x, x_batch)
                gcn1_alpha = self.gcn1_alpha_eval
                gcn2_alpha = self.gcn2_alpha_eval
                gcn3_alpha = self.gcn3_alpha_eval

                # get predictions
                y_score, y_edge_rel_score = self.get_predictions(y, y_edge_index)
                # split beams to separate trees and evaluate joint probability of trees
                beam_results = self.split_beams_to_separate_trees(
                    y_score, y, y_batch, y_beam, y_edge_index, y_edge_type, y_edge_rel_score,
                    calc_joint_prob=True, sort_by='total')

                y = beam_results[0]['y']
                y_batch = beam_results[0]['y_batch']
                y_edge_index = beam_results[0]['y_edge_index']
                y_edge_type = beam_results[0]['y_edge_type']
                y_score = beam_results[0]['y_score']
                y_edge_rel_score = beam_results[0]['y_edge_rel_score']
            else:
                y, y_batch, y_edge_index, y_edge_type = self.gen_graph_greedy(x, x_batch)
                y_score, y_edge_rel_score = self.get_predictions(y, y_edge_index)
                gcn1_alpha = self.gcn1_alpha_eval
                gcn2_alpha = self.gcn2_alpha_eval
                gcn3_alpha = self.gcn3_alpha_eval

        return y, y_batch, y_edge_index, y_edge_type, y_score, y_edge_rel_score, gcn1_alpha, gcn2_alpha, gcn3_alpha

    def get_predictions(self, y, y_edge_index):
        # predictions for nodes from output graph
        y_score = self.lin_node_out(y)
        # build output graph edge features by concatenating corresponding nodes features
        y_edge_features = y[y_edge_index].permute(1, 0, 2)
        y_edge_features = y_edge_features.flatten(1)
        # predictions for edges from output graph
        y_edge_rel_score = self.lin_edge_out(y_edge_features)

        return y_score, y_edge_rel_score

    def train_forward(self, x, x_batch, tgt_y, tgt_edge_index, tgt_edge_type, tgt_y_batch):
        # create tgt nodes embeddings
        y = self.embeds(tgt_y.unsqueeze(1))
        y = F.dropout(y, p=self.emb_dropout_p, training=self.training)
        # remove dimension added by embedding layer
        y = y.squeeze(1)
        # rename to be consistent with evaluation time
        y_edge_index = tgt_edge_index
        y_edge_type = tgt_edge_type
        y_batch = tgt_y_batch
        # store init y embeds for attention purposes
        y_init = y
        # gcn layers
        y, gcn1_alpha = self.decBlock1(x, y, y_edge_index, y_edge_type, x_batch, y_batch, y_init)
        y, gcn2_alpha = self.decBlock2(x, y, y_edge_index, y_edge_type, x_batch, y_batch, y_init)
        y, gcn3_alpha = self.decBlock3(x, y, y_edge_index, y_edge_type, x_batch, y_batch, y_init)

        return y, y_edge_index, gcn1_alpha, gcn2_alpha, gcn3_alpha

    def gen_graph_beam(self, x, x_batch):
        # get beam size to generate all SLT trees at once
        beam_width = torch.tensor(self.beam_width, dtype=torch.long)
        # holds the state of graph nodes as it should look before gcn processing
        y_init = torch.tensor([], dtype=torch.double).to(self.device)
        # holds the state of graph nodes as it should look after gcn processing
        y = torch.tensor([], dtype=torch.double).to(self.device)
        # generates zero for each element - is here so that attention to source graph features would work
        y_batch = torch.tensor([], dtype=torch.long).to(self.device)
        y_beam = torch.tensor([], dtype=torch.long).to(self.device)
        y_eindex = torch.tensor([[], []], dtype=torch.long).to(self.device)
        y_etype = torch.zeros(0, dtype=torch.long).to(self.device)
        pa_ids, gp_ids, ls_ids = [None] * beam_width, [None] * beam_width, [None] * beam_width
        gen_tree = [True] * beam_width
        y_init, y, y_batch, y_beam, y_eindex, y_etype, _, _ = self.gen_subtree_beam(x, x_batch, y_init, y, y_batch, y_beam,
                                                                                    y_eindex, y_etype, gen_tree, pa_ids,
                                                                                    gp_ids, ls_ids)

        return y, y_batch, y_beam, y_eindex, y_etype

    def create_edge(self, y_eindex, y_etype, src_id, tgt_id, etype):
        edge = torch.tensor([[src_id], [tgt_id]], dtype=torch.long).to(self.device)
        edge_type = torch.tensor([etype], dtype=torch.long).to(self.device)
        y_eindex = torch.cat([y_eindex, edge], dim=1)
        y_etype = torch.cat([y_etype, edge_type], dim=0)
        return y_eindex, y_etype

    def gen_subtree_beam(self, x, x_batch, y_init, y, y_batch, y_beam, y_eindex, y_etype, gen_tree, pa_ids, gp_ids, ls_ids):
        # create new nodes for each batch that is not finished yet
        beam_idx_unfinisned = torch.tensor(list(compress(range(len(gen_tree)), gen_tree)), dtype=torch.long).to(self.device)
        y_init_new = torch.zeros((len(beam_idx_unfinisned), self.in_size), dtype=torch.double).to(self.device)
        y_new = torch.zeros((len(beam_idx_unfinisned), self.emb_size), dtype=torch.double).to(self.device)
        y_new_idx = torch.tensor([y.shape[0] + order for order in range(y_new.shape[0])], dtype=torch.long).to(self.device)
        y_init = torch.cat([y_init, y_init_new], dim=0)

        # previous generation trees state
        y_before = y
        y_before_beam = y_beam

        y = torch.cat([y, y_new], dim=0)
        y_batch = torch.cat([y_batch, torch.zeros(beam_idx_unfinisned.shape, dtype=torch.long).to(self.device)], dim=0)
        y_beam = torch.cat([y_beam, beam_idx_unfinisned], dim=0)

        # connect nodes to graph
        for i, i_beam in enumerate(beam_idx_unfinisned):
            y_eindex, y_etype = self.create_edge(y_eindex, y_etype, y_new_idx[i], y_new_idx[i],
                                                 SltEdgeTypes.CURRENT_CURRENT)
            if pa_ids[i_beam] is not None:
                y_eindex, y_etype = self.create_edge(y_eindex, y_etype, pa_ids[i_beam], y_new_idx[i],
                                                     SltEdgeTypes.PARENT_CHILD)
            if gp_ids[i_beam] is not None:
                y_eindex, y_etype = self.create_edge(y_eindex, y_etype, gp_ids[i_beam], y_new_idx[i],
                                                     SltEdgeTypes.GRANDPARENT_GRANDCHILD)
            if ls_ids[i_beam] is not None:
                y_eindex, y_etype = self.create_edge(y_eindex, y_etype, ls_ids[i_beam], y_new_idx[i],
                                                     SltEdgeTypes.LEFTBROTHER_RIGHTBROTHER)
        # process with GCNs
        y_processed = torch.clone(y_init)
        y_processed, gcn1_alpha = self.decBlock1(x, y_processed, y_eindex, y_etype, x_batch, y_batch, y_init)
        y_processed, gcn2_alpha = self.decBlock2(x, y_processed, y_eindex, y_etype, x_batch, y_batch, y_init)
        y_processed, gcn3_alpha = self.decBlock3(x, y_processed, y_eindex, y_etype, x_batch, y_batch, y_init)
        self.gcn1_alpha_eval = gcn1_alpha
        self.gcn2_alpha_eval = gcn2_alpha
        self.gcn3_alpha_eval = gcn3_alpha
        # save attention coefficients - for analysis purposes only
        # the ones from last node generation will be returned
        # update value only for newly created node
        y[y_new_idx] = y_processed[y_new_idx]

        # ------------------------------------- BAYES START

        y_score = self.lin_node_out(y)
        y_prob = F.softmax(y_score, dim=1)

        # from each beam get three most probable tokens
        y_new_prob = y_prob[y_new_idx]
        if y_before.shape[0] == 0:
            # if this is the first generated node (root) - then assign each beam one of the most probable nodes
            _, y_beam_new_best_tokens = torch.topk(y_new_prob, self.beam_width, dim=1, sorted=False)
            y_beam_new_best_tokens = y_beam_new_best_tokens[0]
        else:
            # if there already is some tree generated - select token for each beam based on Bayes rule
            _, y_new_beam_tokens = torch.topk(y_new_prob, self.beam_width, dim=1, sorted=False)
            # merge the tokens into one list
            y_new_tokens, y_new_tokens_counts = torch.unique(y_new_beam_tokens, return_counts=True)
            # count prior probabilities proportional to number of beams in which it was among beam-size best
            y_new_tokens_prior = y_new_tokens_counts / torch.sum(y_new_tokens_counts)

            # calculate likelihoods for each beam tree without the newly generated tokens
            y_before_beam_max, _ = y_before.max(dim=1)
            y_beam_without_new_likelihood = torch.zeros(y_new_idx.shape[0], dtype=torch.double).to(self.device)
            for i, i_beam in enumerate(beam_idx_unfinisned):
                beam_mask = (y_before_beam == i_beam).bool().to(self.device)
                y_before_beam_i = torch.masked_select(y_before_beam_max, beam_mask)
                y_beam_without_new_likelihood[i] = torch.prod(y_before_beam_i)

            # calculate likelihood for each beam tree and each token
            y_beam_without_new_likelihood = y_beam_without_new_likelihood.repeat(y_new_tokens.shape[0], 1).t()
            y_beam_new_tokens_prob = y_prob[y_new_idx][:, y_new_tokens]
            y_beam_with_new_likelihood = torch.mul(y_beam_without_new_likelihood, y_beam_new_tokens_prob)
            y_beam_joint_probs = y_beam_with_new_likelihood * y_new_tokens_prior
            # calculate evidence for each beam
            y_beam_evidence = torch.sum(y_beam_joint_probs, dim=1)
            # calculate posterior for each beam and new token
            y_beam_new_tokens_posterior = (y_beam_joint_probs.t() / y_beam_evidence).t()
            y_beam_new_tokens_posterior_argmax = torch.argmax(y_beam_new_tokens_posterior, dim=1)
            y_beam_new_best_tokens = y_new_tokens[y_beam_new_tokens_posterior_argmax]

        # ------------------------------------- BAYES END

        y_beam_best_embeds = self.embeds(y_beam_new_best_tokens.unsqueeze(1))
        y_init[y_new_idx] = y_beam_best_embeds.squeeze(1)

        y_new_idx_per_batch = [None] * len(pa_ids)
        this_leaf = [False] * len(pa_ids)
        # determine for which batch items the generating process should continue
        # and create list of current node ids per batch item
        gen_subtree = list(gen_tree)
        for i, i_beam in enumerate(beam_idx_unfinisned):
            if y_beam_new_best_tokens[i] == self.end_node_token_id:
                gen_subtree[i_beam] = False
                this_leaf[i_beam] = True
            y_new_idx_per_batch[i_beam] = y_new_idx[i]
        # stop batch items tree generation for those batch items, whose trees reached maximal nodes count
        beam_items_nodes_counts = torch.unique(y_beam, return_counts=True, sorted=True)[1]
        beam_items_nodes_limit_reached = (beam_items_nodes_counts > self.max_output_graph_size)
        gen_subtree = [True if gen_subtree_i and not beam_items_nodes_limit_reached[i] else False
                       for i, gen_subtree_i in enumerate(gen_subtree)]
        if not any(gen_subtree):
            # end leaf node generated for all batch items
            return y_init, y, y_batch, y_beam, y_eindex, y_etype, y_new_idx_per_batch, this_leaf
        else:
            subl_ls_ids = [None] * len(ls_ids)
            while any(gen_subtree):
                # generate sublevel with subtree for each node, until end node is generated
                y_init, y, y_batch, y_beam, y_eindex, y_etype, subl_ls_ids, last_leaf = \
                    self.gen_subtree_beam(x, x_batch, y_init, y, y_batch, y_beam, y_eindex, y_etype, gen_subtree,
                                          y_new_idx_per_batch, pa_ids, subl_ls_ids)
                # stop batch items tree generation for those batch items, whose trees reached maximal nodes count
                beam_items_nodes_counts = torch.unique(y_beam, return_counts=True, sorted=True)[1]
                beam_items_nodes_limit_reached = (beam_items_nodes_counts > self.max_output_graph_size)
                # update for which batch items the subtree generation shall continue
                gen_subtree = [
                    True if gen_subtree_i and not last_leaf[i] and not beam_items_nodes_limit_reached[i] else False for
                    i, gen_subtree_i in enumerate(gen_subtree)]
            return y_init, y, y_batch, y_beam, y_eindex, y_etype, y_new_idx_per_batch, this_leaf

    def split_beams_to_separate_trees(self, y_score, y, y_batch, y_beam, y_edge_index, y_edge_type, y_edge_rel_score,
                                      calc_joint_prob=True, sort_by='total'):
        beam_ids = torch.unique(y_beam, sorted=True)
        beams = []

        sum_joint_tokens_certainty = 0
        sum_joint_edges_certainty = 0
        sum_joint_total_certainty = 0

        for beam_id in beam_ids:
            y_beam_ids = (y_beam == beam_id).nonzero(as_tuple=True)[0]
            y_score_beam = y_score[y_beam_ids]
            y_beam_nodes = y[y_beam_ids]
            y_batch_beam = y_batch[y_beam_ids]
            y_edge_index_beam_ids = [i for i, src in enumerate(y_edge_index[0]) if
                                     src in y_beam_ids and y_edge_index[1][i] in y_beam_ids]
            y_edge_index_beam = y_edge_index.t()[y_edge_index_beam_ids].t()
            # shift edge index ids to the new ones
            y_edge_index_beam[0] = (y_edge_index_beam[0].view(-1, 1) == y_beam_ids).int().argmax(dim=1)
            y_edge_index_beam[1] = (y_edge_index_beam[1].view(-1, 1) == y_beam_ids).int().argmax(dim=1)
            y_edge_type_beam = y_edge_type[y_edge_index_beam_ids]
            y_edge_rel_score_beam = y_edge_rel_score[y_edge_index_beam_ids]

            beam_dict = {
                'y_score': y_score_beam,
                'y': y_beam_nodes,
                'y_batch': y_batch_beam,
                'y_edge_index': y_edge_index_beam,
                'y_edge_type': y_edge_type_beam,
                'y_edge_rel_score': y_edge_rel_score_beam
            }

            if calc_joint_prob:
                # tokens certainty
                tokens_certainty = F.softmax(y_score_beam, dim=1)
                max_tokens_certainty, _ = tokens_certainty.max(dim=1)
                joint_tokens_certainty = torch.prod(max_tokens_certainty)
                mean_tokens_certainty = torch.mean(max_tokens_certainty)
                beam_dict['joint_tokens_certainty'] = joint_tokens_certainty
                beam_dict['mean_tokens_certainty'] = mean_tokens_certainty
                sum_joint_edges_certainty += joint_tokens_certainty
                # edge certainty
                pc_edges_indices = ((y_edge_type_beam == SltEdgeTypes.PARENT_CHILD).nonzero(as_tuple=True)[0])
                pc_edge_rel_score_beam = y_edge_rel_score_beam[pc_edges_indices]
                edges_certainty = F.softmax(y_edge_rel_score_beam, dim=1)
                max_edges_certainty, _ = edges_certainty.max(dim=1)
                joint_edges_certainty = torch.prod(max_edges_certainty)
                mean_edge_certainty = torch.mean(max_edges_certainty)
                beam_dict['joint_edges_certainty'] = joint_edges_certainty
                beam_dict['mean_edges_certainty'] = mean_edge_certainty
                sum_joint_edges_certainty += joint_edges_certainty
                # total certainty
                joint_total_certainty = joint_tokens_certainty * joint_edges_certainty
                beam_dict['joint_total_certainty'] = joint_total_certainty
                mean_total_certainty = torch.mean(torch.cat([max_tokens_certainty, max_edges_certainty]))
                beam_dict['mean_total_certainty'] = mean_total_certainty
                sum_joint_total_certainty += joint_total_certainty

                # latex, latex_symbols = SltParser.slt_to_latex(
                #     self.tokenizer, torch.argmax(y_score_beam, dim=1).cpu(),
                #     torch.argmax(y_edge_rel_score_beam, dim=1).cpu(), y_edge_index_beam.cpu(), y_edge_type_beam.cpu()
                # )
                # print(latex_symbols)
                # print(latex)
                # print(joint_total_certainty)
                # print(mean_total_certainty)
                # print("\n")

            beams.append(beam_dict)

        if calc_joint_prob:
            for beam in beams:
                beam['joint_tokens_certainty'] /= sum_joint_tokens_certainty
                beam['joint_edges_certainty'] /= sum_joint_edges_certainty
                beam['joint_total_certainty'] /= sum_joint_total_certainty

            if sort_by not in ['total', 'tokens', 'edges']:
                sort_by = 'total'
            sort_by = 'joint_' + sort_by + '_certainty'
            beams.sort(key=operator.itemgetter(sort_by), reverse=True)

        return beams

    def gen_graph_greedy(self, x, x_batch):
        # get batch size to generate all SLT trees at once
        bs = torch.unique(x_batch).shape[0]
        # holds the state of graph nodes as it should look before gcn processing
        y_init = torch.tensor([], dtype=torch.double).to(self.device)
        # holds the state of graph nodes as it should look after gcn processing
        y = torch.tensor([], dtype=torch.double).to(self.device)
        y_batch = torch.tensor([], dtype=torch.long).to(self.device)
        y_eindex = torch.tensor([[], []], dtype=torch.long).to(self.device)
        y_etype = torch.zeros(0, dtype=torch.long).to(self.device)
        pa_ids, gp_ids, ls_ids = [None] * bs, [None] * bs, [None] * bs
        gen_tree = [True] * bs
        y_init, y, y_batch, y_eindex, y_etype, _, _ = self.gen_subtree_greedy(x, x_batch, y_init, y, y_batch, y_eindex, y_etype, gen_tree, pa_ids, gp_ids, ls_ids)
        return y, y_batch, y_eindex, y_etype

    def gen_subtree_greedy(self, x, x_batch, y_init, y, y_batch, y_eindex, y_etype, gen_tree, pa_ids, gp_ids, ls_ids):
        # create new nodes for each batch that is not finished yet
        batch_idx_unfinisned = torch.tensor(list(compress(range(len(gen_tree)), gen_tree)), dtype=torch.long).to(self.device)
        y_init_new = torch.zeros((len(batch_idx_unfinisned), self.in_size), dtype=torch.double).to(self.device)
        y_new = torch.zeros((len(batch_idx_unfinisned), self.emb_size), dtype=torch.double).to(self.device)
        y_new_idx = torch.tensor([y.shape[0] + order for order in range(y_new.shape[0])], dtype=torch.long).to(self.device)
        y_init = torch.cat([y_init, y_init_new], dim=0)
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
        y_processed, gcn1_alpha = self.decBlock1(x, y_processed, y_eindex, y_etype, x_batch, y_batch, y_init)
        y_processed, gcn2_alpha = self.decBlock2(x, y_processed, y_eindex, y_etype, x_batch, y_batch, y_init)
        y_processed, gcn3_alpha = self.decBlock3(x, y_processed, y_eindex, y_etype, x_batch, y_batch, y_init)
        self.gcn1_alpha_eval = gcn1_alpha
        self.gcn2_alpha_eval = gcn2_alpha
        self.gcn3_alpha_eval = gcn3_alpha
        # save attention coefficients - for analysis purposes only
        # the ones from last node generation will be returned
        # update value only for newly created node
        y[y_new_idx] = y_processed[y_new_idx]
        # decode newly generated node
        y_new_score = self.lin_node_out(y[y_new_idx])
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
        # stop batch items tree generation for those batch items, whose trees reached maximal nodes count
        batch_items_nodes_counts = torch.unique(y_batch, return_counts=True, sorted=True)[1]
        batch_items_nodes_limit_reached = (batch_items_nodes_counts > self.max_output_graph_size)
        gen_subtree = [True if gen_subtree_i and not batch_items_nodes_limit_reached[i] else False
                       for i, gen_subtree_i in enumerate(gen_subtree)]
        if not any(gen_subtree):
            # end leaf node generated for all batch items
            return y_init, y, y_batch, y_eindex, y_etype, y_new_idx_per_batch, this_leaf
        else:
            subl_ls_ids = [None] * len(ls_ids)
            while any(gen_subtree):
                # generate sublevel with subtree for each node, until end node is generated
                y_init, y, y_batch, y_eindex, y_etype, subl_ls_ids, last_leaf = \
                    self.gen_subtree_greedy(x, x_batch, y_init, y, y_batch, y_eindex, y_etype, gen_subtree, y_new_idx_per_batch, pa_ids, subl_ls_ids)
                # stop batch items tree generation for those batch items, whose trees reached maximal nodes count
                batch_items_nodes_counts = torch.unique(y_batch, return_counts=True, sorted=True)[1]
                batch_items_nodes_limit_reached = (batch_items_nodes_counts > self.max_output_graph_size)
                # update for which batch items the subtree generation shall continue
                gen_subtree = [True if gen_subtree_i and not last_leaf[i] and not batch_items_nodes_limit_reached[i] else False for i, gen_subtree_i in enumerate(gen_subtree)]
            return y_init, y, y_batch, y_eindex, y_etype, y_new_idx_per_batch, this_leaf

