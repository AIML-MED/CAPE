import torch
import torch.nn as nn
from torch.nn import init
import torchvision.models as models
from torch.nn import functional as F
import copy, sys, cv2
import numpy as np
import sys



def weight_init_kaiming(m):
    class_names = m.__class__.__name__
    if class_names.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif class_names.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
        #init.constant_(m.bias.data, 0.0)
        


class Temperature(nn.Module):

    def __init__(self, max_scale=8):
        super(Temperature, self).__init__()
        self.T = nn.Parameter(torch.zeros(1))
        self.max_scale = max_scale

    def forward(self, x):
        return x / torch.exp(self.T)


class Temperature3(nn.Module):

    def __init__(self, max_scale=8):
        super(Temperature3, self).__init__()
        self.T = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.max_scale = max_scale

    def forward(self, x):
        return x / torch.exp(self.T)


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        if self.config['pretrained']:
            net = getattr(models, self.config['network'])(weights='IMAGENET1K_V1')
        else:
            net = getattr(models, self.config['network'])()

        # resnet50
        in_features = list(net.fc.modules())[-1].in_features
        modules = list(net.children())
        modules = modules[:-2]
        self.net = nn.Sequential(*modules)

        # classification module
        self.classifiers = nn.ModuleDict()
        for t_name in self.config['cls_cols_dict']:
            modules = list()
            modules.append(nn.AdaptiveAvgPool2d((1,1)))
            modules.append(nn.Flatten())
            modules.append(nn.Linear(in_features, self.config['cls_cols_dict'][t_name]))
            classifier = nn.Sequential(*modules)
            classifier.apply(weight_init_kaiming)
            self.classifiers[t_name] = classifier

        self.t1 = Temperature()  # for contribution
        self.t2 = Temperature()  # for weight
        self.t3 = Temperature3()  # for both

    def forward(self, input):
        features = self.net(input)
        cls_maps = list()
        for c_name in self.classifiers:
            classifier = self.classifiers[c_name]
            if c_name == 'cape':
                c = self.get_cam_faster(features.detach(), classifier)
            else:
                c = self.get_cam_faster(features, classifier)
            cls_maps.append(c)

        output = {}
        for c_idx, c_name in enumerate(self.classifiers):
            output[c_name] = self.contribution_calculator(cls_maps[c_idx], cls_maps[c_idx], c_name)
            output[c_name]['cls_map'] = cls_maps[c_idx]
        return output

        
    def get_cam_faster(self, features, classifier): 
        cls_weights = classifier[-1].weight 
        cls_bias = classifier[-1].bias

        act_maps = F.conv2d(features, cls_weights.view(cls_weights.shape[0], cls_weights.shape[1], 1, 1),
                            cls_bias, stride=1, padding=0, dilation=1, groups=1)

        return act_maps

    def contribution_calculator(self, pixel_values, weight, c_name='cape'):
        _weight = weight
        if c_name == 'cape':
            contribution = self.t1(pixel_values).view(pixel_values.shape[0], pixel_values.shape[1], -1)
            weight = self.t2(_weight).mean(dim=1, keepdim=True).view(pixel_values.shape[0], 1, -1)
            divisor = contribution + weight
            denominator = contribution.unsqueeze(-1) + weight.unsqueeze(2)
            normalizer = denominator.max(dim=-1, keepdims=True)[0].max(dim=1, keepdims=True)[0]
            denominator = denominator - normalizer
            divisor = divisor - normalizer.squeeze(-1)
            
            weighted_contribution = divisor.exp() / denominator.exp().sum(dim=3).sum(dim=1, keepdims=True)
            weighted_contribution = weighted_contribution.view(pixel_values.shape)
            outcome = weighted_contribution.sum(dim=(2, 3))

            denominator_soft = self.t3(denominator)
            divisor_soft = self.t3(divisor.unsqueeze(-1)).squeeze(-1)
            weighted_contribution_soft = divisor_soft.exp() / denominator_soft.exp().sum(dim=3).sum(dim=1, keepdims=True)
            weighted_contribution_soft = weighted_contribution_soft.view(pixel_values.shape)
            outcome_soft = weighted_contribution_soft.sum(dim=(2, 3))

            logcampe_clip0 = (divisor + normalizer.squeeze(-1)).view(-1,
                                weighted_contribution.shape[1], weighted_contribution.shape[2], 
                                weighted_contribution.shape[3])
            logcampe_clip0 = logcampe_clip0.clip(0)

        elif c_name == 'orig':
            contribution = pixel_values
            weight = torch.ones([contribution.shape[0], 1] + list(contribution.shape[2:]), device=contribution.device) / (
                        contribution.shape[2] * contribution.shape[3])
            weighted_contribution = contribution * weight
            outcome = weighted_contribution.sum(dim=(2, 3))
            outcome_soft = outcome.softmax(dim=1)
            logcampe_clip0 = weighted_contribution

        else:
            raise ValueError('Undefined c_name')
        
        return {
            'outcome': outcome,
            'outcome_soft': outcome_soft,
            'weighted_contribution': weighted_contribution,
            'logcampe_clip0': logcampe_clip0,
        }


