import torch
import torch.nn.functional as F


def loss_termination(logits, gt, termination_token):
    termination_tokens_mask = (gt == termination_token).to(torch.long)
    ce_loss = F.cross_entropy(logits, gt, reduction='none')
    ce_loss_masked = ce_loss * termination_tokens_mask
    ce_loss_masked = torch.mean(ce_loss_masked)
    return ce_loss_masked
