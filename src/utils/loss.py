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


def masked_log_softmax(vector, mask, dim=-1):
    if mask is not None:
        mask = mask.float()
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


def masked_softmax(vector, mask, dim=-1, mask_fill_value=float('-inf')):
    mask = mask.float()
    masked_vector = vector.masked_fill((1 - mask).bool(), mask_fill_value)
    result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result

def calculate_loss(out, end_node_token_id, device, writer, writer_idx=0):
    # calculate loss for output graph node predictions
    loss_out_node = F.cross_entropy(out.y_score, out.tgt_y)

    # calculate additional loss penalizing classification non-end nodes as end nodes
    loss_end_nodes = loss_termination(out.y_score, out.tgt_y, end_node_token_id)

    # calculate loss for source graph node predictions
    # train to predict symbol from component
    x_gt_node = out.attn_gt.argmax(dim=0)
    x_gt = out.tgt_y[x_gt_node]
    loss_enc_nodes = F.cross_entropy(out.x_score, x_gt)

    # calculate loss for source graph edge predictions
    if out.edge_type.shape[0] > 0:
        loss_enc_edges = F.cross_entropy(out.edge_type_score, out.edge_type)
    else:
        loss_enc_edges = torch.tensor(0, dtype=torch.double)

    # compute loss for components classification
    loss_comp_class = F.cross_entropy(out.comp_class, x_gt)

    # # calculate loss for attention to source graph
    # average attentions in all decoder GCN layers
    alpha_batch_mask = (out.y_batch.unsqueeze(1) - out.x_batch.unsqueeze(0) == 0)
    block1_attn = out.gcn1_alpha
    block2_attn = out.gcn1_alpha
    block3_attn = out.gcn1_alpha
    attn_gt = out.attn_gt * 100

    # # no_end_node_indices = (out.tgt_y != end_node_token_id).long()
    # # no_end_node_mask = no_end_node_indices.unsqueeze(1).repeat(1, out.x.shape[0])

    # attn_gt = masked_log_softmax(attn_gt, alpha_batch_mask, dim=1)
    # block1_attn = masked_log_softmax(block1_attn, alpha_batch_mask, dim=1)
    # block2_attn = masked_log_softmax(block2_attn, alpha_batch_mask, dim=1)
    # block3_attn = masked_log_softmax(block3_attn, alpha_batch_mask, dim=1)
    # loss_block1_attn = F.kl_div(block1_attn, attn_gt, reduction='sum', log_target=True)
    # loss_block2_attn = F.kl_div(block2_attn, attn_gt, reduction='sum', log_target=True)
    # loss_block3_attn = F.kl_div(block3_attn, attn_gt, reduction='sum', log_target=True)

    attn_gt = masked_softmax(attn_gt, alpha_batch_mask, dim=1)
    loss_block1_attn = F.mse_loss(block1_attn, attn_gt, reduction='mean')
    loss_block2_attn = F.mse_loss(block2_attn, attn_gt, reduction='mean')
    loss_block3_attn = F.mse_loss(block3_attn, attn_gt, reduction='mean')

    loss_block_attn_mean = torch.mean(torch.stack([loss_block1_attn, loss_block2_attn, loss_block3_attn]))

    # calculate loss for output graph SRT edge type predictions - take in account only parent-child edges
    tgt_edge_pc_indices = ((out.tgt_edge_type == SltEdgeTypes.PARENT_CHILD).nonzero(as_tuple=True)[0])
    tgt_pc_edge_relation = out.tgt_edge_relation[tgt_edge_pc_indices]
    out_pc_edge_relation = out.y_edge_rel_score[tgt_edge_pc_indices]
    loss_out_edge = F.cross_entropy(out_pc_edge_relation, tgt_pc_edge_relation)

    # losses_out_edge = F.cross_entropy(out.y_edge_rel_score, out.tgt_edge_relation, reduction='none')
    # tgt_edge_pc_mask = (out.tgt_edge_type == SltEdgeTypes.PARENT_CHILD).long()
    # losses_out_edge = losses_out_edge * tgt_edge_pc_mask
    # loss_out_edge = torch.mean(losses_out_edge)

    if writer:
        writer.add_scalar('ItemLossOutNode/train', loss_out_node.item(), writer_idx)
        writer.add_scalar('ItemLossOutEdge/train', loss_out_edge.item(), writer_idx)
        writer.add_scalar('ItemLossVggClass/train', loss_comp_class.item(), writer_idx)
        writer.add_scalar('ItemLossEncNode/train', loss_enc_nodes.item(), writer_idx)
        writer.add_scalar('ItemLossEncEdge/train', loss_enc_edges.item(), writer_idx)
        writer.add_scalar('ItemLossAttn/train', loss_block_attn_mean.item(), writer_idx)
        writer.add_scalar('ItemLossLeafNode/train', loss_end_nodes.item(), writer_idx)

    # print(loss_out_node)
    # print(loss_out_edge)
    # print(loss_comp_class)
    # print(loss_enc_nodes)
    # print(loss_enc_edges)
    # print(loss_block_attn_mean)
    # print(loss_end_nodes)

    loss = \
        loss_out_node + \
        loss_out_edge + \
        0.5 * loss_comp_class + \
        0.5 * loss_enc_nodes + \
        0.5 * loss_enc_edges + \
        0.5 * loss_end_nodes + \
        0.6 * loss_block_attn_mean
    return loss
