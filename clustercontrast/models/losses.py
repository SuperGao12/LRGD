import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np
from IPython import embed


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes=0, epsilon=0.1, topk_smoothing=False):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()
        self.k = 1 if not topk_smoothing else self.num_classes // 50

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        if self.k > 1:
            topk = torch.argsort(-log_probs)[:, :self.k]
            targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1 - self.epsilon)
            targets += torch.zeros_like(log_probs).scatter_(1, topk, self.epsilon / self.k)
        else:
            targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
            targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss





##################
def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [B, m, d]
      y: pytorch Variable, with shape [B, n, d]
    Returns:
      dist: pytorch Variable, with shape [B, m, n]
    """
    B = x.size(0)
    m, n = x.size(1), y.size(1)
    x_norm = torch.pow(x, 2).sum(2, keepdim=True).sqrt().expand(B, m, n)
    y_norm = torch.pow(y, 2).sum(2, keepdim=True).sqrt().expand(B, n, m).transpose(-2, -1)
    xy_intersection = x @ y.transpose(-2, -1)
    dist = xy_intersection / (x_norm * y_norm)
    return torch.abs(dist)


class Dissimilar(object):
    def __init__(self, dynamic_balancer=True):
        self.dynamic_balancer = dynamic_balancer

    def __call__(self, features):
        B, N, C = features.shape
        dist_mat = cosine_dist(features, features)  # B*N*N
        # dist_mat = euclidean_dist(features, features)
        # 上三角index
        top_triu = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        _dist = dist_mat[:, top_triu]

        # 1.用softmax替换平均，使得相似度更高的权重更大
        if self.dynamic_balancer:
            weight = F.softmax(_dist, dim=-1)
            dist = torch.mean(torch.sum(weight * _dist, dim=1))
        # 2.直接平均
        else:
            dist = torch.mean(_dist, dim=(0, 1))
        return dist
########################



class WeightedCrossEntropyLoss(nn.Module):
    """
    加权的交叉熵损失类
    """

    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input, target, target_cross,reduction='mean'):
            #############################
            #############################
            #############################
            #############################
            #############################
            #############################
            #############################
            #############################

            return 0



class KL_loss(nn.Module):
    def __init__(self):
        super(KL_loss, self).__init__()


    def forward(self, f_out_rgb, f_out_ir, memory_rgb_hat, memory_ir_hat, r2i, i2r):
        #############################
        #############################
        #############################
        #############################
        #############################
        #############################
        #############################
        #############################
        return 0




class Adapt_Fusion(torch.nn.Module):
    def __init__(self, lambda_val=0.9, T1=0.05):
        super(Adapt_Fusion, self).__init__()


    def compute_cross_attention(self, query, key, value):
        #############################
        #############################
        #############################
        #############################
        #############################
        #############################
        #############################
        #############################

        return 0


class AdaptiveContrastiveLoss(torch.nn.Module):
    def __init__(self,temp=0.2):
        super(AdaptiveContrastiveLoss, self).__init__()

    def forward(self, feat_rgb, feat_rgb_a):
        #############################
        #############################
        #############################
        #############################
        #############################
        #############################
        #############################
        #############################
        return 0