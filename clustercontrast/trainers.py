from __future__ import print_function, absolute_import
from audioop import cross
import time
from .utils.meters import AverageMeter
import torch.nn as nn
import torch
from torch.nn import functional as F
import math
import numpy as np
import os
from .models.losses import KL_loss,Adapt_Fusion,AdaptiveContrastiveLoss


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

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x



class ClusterContrastTrainer_DCL(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer_DCL, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        # 计算损失
        self.memory_rgb = memory
        self.dissimilar = Dissimilar(dynamic_balancer=True)

    def train(self, epoch, data_loader_ir, data_loader_rgb, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()
        # 将模型设置为训练模式。
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        # 分别用于记录每个批次的训练时间、数据加载时间和损失值。AverageMeter 通常用于计算和存储平均值和当前值。
        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()

            data_time.update(time.time() - end)

            # process inputs
            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb, inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)

            # forward
            inputs_rgb = torch.cat((inputs_rgb, inputs_rgb1), 0)
            labels_rgb = torch.cat((labels_rgb, labels_rgb), -1)
            _, f_out_rgb, f_out_ir, SDC_rgb, SDC_ir = self._forward(inputs_rgb, inputs_ir, modal=0)

            loss_ir = self.memory_ir(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)

            #
            Dissimilar_LOSS_ir = self.dissimilar(torch.stack(SDC_ir, dim=1))
            Dissimilar_LOSS_rgb = self.dissimilar(torch.stack(SDC_rgb, dim=1))
            Dissimilar_LOSS = Dissimilar_LOSS_ir + Dissimilar_LOSS_rgb

            loss = loss_ir + loss_rgb + Dissimilar_LOSS

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      'Dissimilar_LOSS {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg, loss_ir, loss_rgb,Dissimilar_LOSS))

    def _parse_data_rgb(self, inputs):
        imgs, imgs1, _, pids, _, indexes = inputs
        return imgs.cuda(), imgs1.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir(self, inputs):
        # imgs, _, _, pids, _, indexes = inputs
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, modal=0,label_1=None, label_2=None):
        return self.encoder(x1, x2, modal=modal )


class ClusterContrastTrainer(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory_ir = memory
        self.memory_rgb = memory
        self.memory_hybrid = memory


        self.dissimilar = Dissimilar(dynamic_balancer=True)
        self.KLloss = KL_loss()
        self.Adapt_Fusion = Adapt_Fusion()
        self.Adaloss = AdaptiveContrastiveLoss()


    def train(self, epoch, data_loader_ir, data_loader_rgb, optimizer, print_freq=10, train_iters=400, i2r=None,
              r2i=None, i2r_p=None, r2i_p=None):

        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_ir = data_loader_ir.next()
            inputs_rgb = data_loader_rgb.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb, inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)

            # forward
            inputs_rgb = torch.cat((inputs_rgb, inputs_rgb1), 0)
            labels_rgb = torch.cat((labels_rgb, labels_rgb), -1)

            _, f_out_rgb, f_out_ir, SDC_rgb, SDC_ir = self._forward(inputs_rgb, inputs_ir, modal=0)


            rgb2ir_labels = torch.tensor([r2i[key.item()] for key in labels_rgb]).cuda()
            ir2rgb_labels = torch.tensor([i2r[key.item()] for key in labels_ir]).cuda()

            rgb2ir_p_labels = torch.tensor([r2i_p[key.item()] for key in labels_rgb]).cuda()
            ir2rgb_p_labels = torch.tensor([i2r_p[key.item()] for key in labels_ir]).cuda()

            Dissimilar_LOSS_ir = self.dissimilar(torch.stack(SDC_ir, dim=1))
            Dissimilar_LOSS_rgb = self.dissimilar(torch.stack(SDC_rgb, dim=1))
            Dissimilar_LOSS = Dissimilar_LOSS_ir + Dissimilar_LOSS_rgb


            f_rgb_updated = self.Adapt_Fusion(f_out_rgb, self.memory_ir.features[rgb2ir_labels], self.memory_ir.features[rgb2ir_labels])
            f_ir_updated = self.Adapt_Fusion(f_out_ir, self.memory_rgb.features[ir2rgb_labels], self.memory_rgb.features[ir2rgb_labels])

            loss_adap_rgb = self.Adaloss(f_out_rgb, f_rgb_updated)
            loss_adap_ir = self.Adaloss(f_out_ir, f_ir_updated)
            loss_adap = loss_adap_rgb+loss_adap_ir


            loss_ir = self.memory_ir(f_out_ir, labels_ir)
            loss_rgb = self.memory_rgb(f_out_rgb, labels_rgb)
            cross_loss = self.memory_rgb(f_out_ir, ir2rgb_labels.long(), ir2rgb_p_labels,
                                         cross='true') + self.memory_ir(f_out_rgb, rgb2ir_labels.long(),
                                                                        rgb2ir_p_labels, cross='true')

            loss_kl = self.KLloss(f_out_rgb, f_out_ir, self.memory_rgb.features, self.memory_ir.features, r2i, i2r)
            # loss = loss_ir + loss_rgb + 1.0 * Dissimilar_LOSS + 1.0 * cross_loss + 0.3 * loss_kl + 0.1 * loss_adap # RegDB
            loss = loss_ir + loss_rgb + 1.0 * Dissimilar_LOSS + 1.0 * cross_loss + 0.3 * loss_kl + 1.0 * loss_adap   # SYSU

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      'Loss cross {:.3f}\t'
                      'Loss kl {:.3f}\t'
                      'loss_adap{:.3f}\t'

                      .format(epoch, i + 1, len(data_loader_rgb),
                              batch_time.val, batch_time.avg, losses.val, losses.avg,
                              loss_ir, loss_rgb, cross_loss, loss_kl, loss_adap))



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())


            batch_time.update(time.time() - end)
            end = time.time()

    def _parse_data_rgb(self, inputs):
        imgs, imgs1, _, pids, indexes, _ = inputs
        return imgs.cuda(), imgs1.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, indexes, _ = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, modal=0):
        return self.encoder(x1, x2, modal)

