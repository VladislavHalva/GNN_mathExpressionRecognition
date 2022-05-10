import torch
import torch.nn.functional as F

from src.definitions.SltEdgeTypes import SltEdgeTypes


def loss_termination(logits, gt, termination_token):
    termination_tokens_mask = (gt == termination_token).to(torch.long)
    ce_loss = F.cross_entropy(logits, gt, reduction='none')
    ce_loss_masked = ce_loss * termination_tokens_mask
    ce_loss_masked = torch.mean(ce_loss_masked)
    return ce_loss_masked


def masked_mse_loss(input, target, mask):
    diff2 = (torch.flatten(input) - torch.flatten(target)) ** 2.0 * torch.flatten(mask)
    result = torch.sum(diff2) / torch.sum(mask)
    return result


def calculate_loss(out, end_node_token_id, device):
    # calculate loss for output graph node predictions
    loss_out_node = F.cross_entropy(out.y_score, out.tgt_y)

    # calculate additional loss penalizing classification non-end nodes as end nodes
    loss_end_nodes = loss_termination(out.y_score, out.tgt_y, end_node_token_id)

    # calculate loss for source graph node predictions
    # train to predict symbol from component
    x_gt_node = out.attn_gt.argmax(dim=0)
    x_gt = out.tgt_y[x_gt_node]
    loss_enc_nodes = F.cross_entropy(out.x_score, x_gt)

    # compute loss for components classification
    loss_comp_class = F.cross_entropy(out.comp_class, x_gt)

    # calculate loss for attention to source graph
    # average attentions in all decoder GCN layers
    gcn_alpha_avg = torch.cat(
        (
            out.gcn1_alpha.unsqueeze(0),
            out.gcn2_alpha.unsqueeze(0),
            out.gcn3_alpha.unsqueeze(0)
        ), dim=0)
    gcn_alpha_avg = torch.mean(gcn_alpha_avg, dim=0)

    # alpha_batch_mask = (out.y_batch.unsqueeze(1) - out.x_batch.unsqueeze(0) == 0).long()
    no_end_node_indices = (out.tgt_y != end_node_token_id).long()
    no_end_node_mask = no_end_node_indices.unsqueeze(1).repeat(1, out.x.shape[0])
    # do not calculate loss for end leaf nodes attention
    gcn_alpha_avg = gcn_alpha_avg * no_end_node_mask

    loss_gcn_alpha_avg = F.mse_loss(
        gcn_alpha_avg.double(),
        out.attn_gt.double()
    ).float()

    # loss_gcn_alpha_avg = masked_mse_loss(
    #     (gcn_alpha_avg),
    #     (out.attn_gt),
    #     alpha_mask
    # )

    # loss_gcn_alpha_avg = F.kl_div(
    #     F.log_softmax(gcn_alpha_avg, dim=1),
    #     F.log_softmax(out.attn_gt, dim=1),
    #     reduction='batchmean', log_target=True
    # )

    # calculate loss for output graph SRT edge type predictions - take in account only parent-child edges
    tgt_edge_pc_indices = ((out.tgt_edge_type == SltEdgeTypes.PARENT_CHILD).nonzero(as_tuple=True)[0])
    tgt_pc_edge_relation = out.tgt_edge_relation[tgt_edge_pc_indices]
    out_pc_edge_relation = out.y_edge_rel_score[tgt_edge_pc_indices]
    loss_out_edge = F.cross_entropy(out_pc_edge_relation, tgt_pc_edge_relation)

    loss = loss_out_node + loss_out_edge + 0.5 * loss_comp_class + 0.5 * loss_enc_nodes + 0.5 * loss_end_nodes + 0.5 * loss_gcn_alpha_avg
    return loss
