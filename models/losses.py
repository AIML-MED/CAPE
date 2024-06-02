import torch
import torch.nn.functional as F
import torch.nn as nn

def target_maker(targets, config):
    start, end = 0, 0
    cls_targets = list()
    for t_name in config['cls_cols_dict']:
        start = end
        end += config['cls_cols_dict'][t_name]
        cls_targets.append(targets[:, start:end])

    return cls_targets

def normalize(loss, weight):
    weight_sum = weight.sum(dim=0)
    num_weight = torch.clamp(weight_sum, 1.)
    loss = (loss * weight).sum(dim=0) / num_weight
    return loss

"""
Cross entropy loss
"""
def ce(logits, targets, c_name):
    if c_name =='cape':
        loss = - targets * torch.log(logits + 1e-10)
    elif c_name == 'orig':
        loss = -targets * torch.log_softmax(logits, dim=1)
    else:
        raise ValueError('Undefined c_name')
    
    loss = loss.sum(dim=1, keepdim=True)
    weighting = (targets[:, 0:1] != 1e-10).type(torch.float)
    loss = normalize(loss, weighting)

    return loss

def classification_loss(config, cls_logits, cls_targets):
    loss_cls = {}
    for c_name, l, t in zip(config['cls_cols_dict'], cls_logits, cls_targets):
        ls = ce(l, t, c_name)
        loss_cls['loss_'+c_name] = ls
    return loss_cls


"""
Distillation Loss
"""
class SoftTargetKDLoss(nn.Module):
    def __init__(self, T=2.0, scale=1.0):
        super(SoftTargetKDLoss, self).__init__()
        self.T = T
        self.scale = scale

    def forward(self, out_student, out_teacher):
        loss = F.kl_div(torch.log(out_student), 
                        F.log_softmax(out_teacher/self.T, dim=1), 
                        reduction='none', log_target=True) * self.scale * self.T * self.T
        return loss.sum(dim=1).mean(dim=0, keepdims=True)
