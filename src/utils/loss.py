import torch
import torch.nn.functional as F

from src.definitions.SltEdgeTypes import SltEdgeTypes


def loss_termination(logits, gt, termination_token):
    termination_tokens_mask = (gt == termination_token).to(torch.long)
    ce_loss = F.cross_entropy(logits, gt, reduction='none')
    ce_loss_masked = ce_loss * termination_tokens_mask
    ce_loss_masked = torch.mean(ce_loss_masked)
    return ce_loss_masked


def calculate_loss(out, end_node_token_id):
    # calculate loss for output graph node predictions
    loss_out_node = F.cross_entropy(out.y_score, out.tgt_y)

    # calculate additional loss penalizing classification non-end nodes as end nodes
    loss_end_nodes = loss_termination(out.y_score, out.tgt_y, end_node_token_id)

    # calculate loss for source graph node predictions
    # train to predict symbol from component
    x_gt_node = out.attn_gt.argmax(dim=0)
    x_gt = out.tgt_y[x_gt_node]
    loss_enc_nodes = F.cross_entropy(out.x_score, x_gt)

    # calculate loss for attention to source graph

    # get mask for batch target and source graph nodes correspondence
    alpha_batch_mask = (out.y_batch.unsqueeze(1) - out.x_batch.unsqueeze(0) == 0).long()
    # transform attention gt so that it matches the desired form of attention
    attn_gt = F.softmax(out.attn_gt.masked_fill((1 - alpha_batch_mask).bool(), float('-inf')), dim=1)
    # transform attention gt with log-softmax so that it suits the needs of KL-div and is numerically stable
    attn_gt = F.log_softmax(attn_gt, dim=1)
    # average attentions in all decoder GCN layers
    gcn_alpha_avg = torch.cat(
        (
            out.gcn1_alpha.unsqueeze(0),
            out.gcn2_alpha.unsqueeze(0),
            out.gcn3_alpha.unsqueeze(0)
        ), dim=0)
    gcn_alpha_avg = torch.mean(gcn_alpha_avg, dim=0)
    gcn_alpha_avg = F.softmax(gcn_alpha_avg.masked_fill((1 - alpha_batch_mask).bool(), float('-inf')), dim=1)
    gcn_alpha_avg = F.log_softmax(gcn_alpha_avg, dim=1)
    # calculate KL divergence itself
    loss_gcn_alpha_avg = F.kl_div(
        gcn_alpha_avg.type(torch.double),
        attn_gt.type(torch.double),
        reduction='sum', log_target=True).type(torch.float)

    # calculate loss for output graph SRT edge type predictions - take in account only parent-child edges
    tgt_edge_pc_indices = ((out.tgt_edge_type == SltEdgeTypes.PARENT_CHILD).nonzero(as_tuple=True)[0])
    tgt_pc_edge_relation = out.tgt_edge_relation[tgt_edge_pc_indices]
    out_pc_edge_relation = out.y_edge_rel_score[tgt_edge_pc_indices]
    loss_out_edge = F.cross_entropy(out_pc_edge_relation, tgt_pc_edge_relation)

    loss = loss_out_node + loss_out_edge + 0.5 * loss_gcn_alpha_avg + 0.5 * loss_enc_nodes + 0.5 * loss_end_nodes
    return loss
