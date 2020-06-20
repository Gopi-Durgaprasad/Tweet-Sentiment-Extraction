import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def dist_between(start_logits, end_logits, device, max_seq_len):

    linear_func = torch.tensor(np.linspace(0,1,max_seq_len, endpoint=False), requires_grad=False)
    linear_func = linear_func.to(device)

    start_pos = (start_logits*linear_func).sum(axis=1)
    end_pos = (end_logits*linear_func).sum(axis=1)

    diff = end_pos-start_pos

    return diff.sum(axis=0)/diff.size(0)


def distloss(start_logits, end_logits, start_positions, end_positions, device, max_seq_len, scale=1):

    start_logits = torch.nn.Softmax(1)(start_logits) # shape ; (batch, max_seq_len)
    end_logits = torch.nn.Softmax(1)(end_logits)

    start_one_hot = torch.nn.functional.one_hot(start_positions, num_classes=max_seq_len).to(device)
    end_one_hot = torch.nn.functional.one_hot(end_positions, num_classes=max_seq_len).to(device)

    pred_dist = dist_between(start_logits, end_logits, device, max_seq_len)
    gt_dist = dist_between(start_one_hot, end_one_hot, device, max_seq_len) 
    diff = (gt_dist-pred_dist)

    rev_diff_squared = 1-torch.sqrt(diff*diff) # as diff is smaller, make it get closer to the one
    loss = -torch.log(rev_diff_squared) # by using negative log function, if argument is near zero -> inifinite, near one -> zero

    return loss*scale

def cross_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    gold = gold.contiguous().view(-1)
    if smoothing:
        eps = 0.1
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred,  requires_grad=False).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean(dim = -1)
    else:
        loss = F.cross_entropy(pred, gold,  reduction='mean')
    return loss

def loss_fn(start_logits, end_logits, start_positions, end_positions,device, max_seq_len):
    loss_fct = nn.CrossEntropyLoss()
    start_loss = cross_loss(start_logits, start_positions)
    end_loss = cross_loss(end_logits, end_positions)
    dist_loss = distloss(start_logits, end_logits,start_positions, end_positions, device, max_seq_len)
    total_loss = start_loss + end_loss + dist_loss
    return total_loss
