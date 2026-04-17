# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta

import ot
from sklearn.cluster import DBSCAN
from PIL import Image
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.trainers import ClusterContrastTrainer,ClusterContrastTrainer_DCL
from clustercontrast.evaluators import Evaluator, extract_features
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor,Preprocessor_color,Preprocessor_aug
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance,compute_ranked_list
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam
import os
import torch.utils.data as data
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import copy
import math
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing,ChannelExchange,Gray
from collections import Counter

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from torchvision.transforms import InterpolationMode
import faiss
from evel import *
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")


def get_data(name, data_dir, trial=0):
    dataset = datasets.create(name, root=data_dir, trial=trial)
    return dataset


def get_train_loader_ir(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False,train_transformer=None,train_transformer1=None):
    
    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    if train_transformer1 is None:
        train_loader = IterLoader(
            DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    else:
        train_loader = IterLoader(
            DataLoader(Preprocessor_aug(train_set, root=dataset.images_dir, transform=train_transformer,
                                        transform1=train_transformer1),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader
    return train_loader


def get_train_loader_color(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False,train_transformer=None,train_transformer1=None):
    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    if train_transformer1 is None:
        train_loader = IterLoader(
            DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    else:
        train_loader = IterLoader(
            DataLoader(Preprocessor_color(train_set, root=dataset.images_dir, transform=train_transformer,transform1=train_transformer1),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None,test_transformer=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    if test_transformer is None:
        test_transformer = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.ToTensor(),
            normalizer
        ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    if args.arch == 'vit_base':
        print('vit_base')
        model = models.create(args.arch,img_size=(args.height,args.width),drop_path_rate=args.drop_path_rate
                , pretrained_path = args.pretrained_path,hw_ratio=args.hw_ratio, conv_stem=args.conv_stem)
    else:
        model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
        
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)#,output_device=1)
    return model


class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (128, 384)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.LANCZOS) # Image.ANTIALIAS
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)


def process_query_sysu(data_path, mode = 'all', relabel=False):
    if mode== 'all':
        ir_cameras = ['cam3','cam6']
    elif mode =='indoor':
        ir_cameras = ['cam3','cam6']
    
    file_path = os.path.join(data_path,'exp/test_id.txt')
    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)


def process_gallery_sysu(data_path, mode = 'all', trial = 0, relabel=False):
    
    random.seed(trial)
    
    if mode== 'all':
        rgb_cameras = ['cam1','cam2','cam4','cam5']
    elif mode =='indoor':
        rgb_cameras = ['cam1','cam2']
        
    file_path = os.path.join(data_path,'exp/test_id.txt')
    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_rgb.append(random.choice(new_files))
    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_rgb:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)
    return gall_img, np.array(gall_id), np.array(gall_cam)
    

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip


def extract_gall_feat(model,gall_loader,ngall):
    pool_dim=768*cls_token_num    ########################768 2048
    net = model
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(gall_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc,_ = net( input,input, 1)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1,_ = net( flip_input,flip_input, 1)
            feature_fc = (feat_fc.detach() + feat_fc_1.detach())/2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            gall_feat_fc[ptr:ptr+batch_num,: ]   = feature_fc.cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat_fc


def extract_query_feat(model,query_loader,nquery):
    pool_dim=768*cls_token_num     ############################# 768 2048
    net = model
    net.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label ) in enumerate(query_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc,_ = net( input, input,2)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1,_ = net( flip_input,flip_input, 2)
            feature_fc = (feat_fc.detach() + feat_fc_1.detach())/2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            query_feat_fc[ptr:ptr+batch_num,: ]   = feature_fc.cpu().numpy()
            
            ptr = ptr + batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return query_feat_fc


def eval_sysu(distmat, q_pids, g_pids, q_camids, g_camids, max_rank = 20):
    """Evaluation with sysu metric
    Key: for each query identity, its gallery images from the same camera view are discarded. "Following the original setting in ite dataset"
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    pred_label = g_pids[indices]
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    # compute cmc curve for each query
    new_all_cmc = []
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0. # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (q_camid == 3) & (g_camids[order] == 2)
        keep = np.invert(remove)
        
        # compute cmc curve
        # the cmc calculation is different from standard protocol
        # we follow the protocol of the author's released code
        new_cmc = pred_label[q_idx][keep]
        new_index = np.unique(new_cmc, return_index=True)[1]
        new_cmc = [new_cmc[index] for index in sorted(new_index)]
        
        new_match = (new_cmc == q_pid).astype(np.int32)
        new_cmc = new_match.cumsum()
        new_all_cmc.append(new_cmc[:max_rank])
        
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(orig_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx]/ (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q   # standard CMC
    
    new_all_cmc = np.asarray(new_all_cmc).astype(np.float32)
    new_all_cmc = new_all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return new_all_cmc, mAP, mINP


def pairwise_distance(features_q, features_g):
    x = torch.from_numpy(features_q)
    y = torch.from_numpy(features_g)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m.numpy()


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    log_s1_name = 'sysu_s1'
    log_s2_name = 'sysu_s2'
    # main_worker_stage1(args,log_s1_name) # Stage 1
    main_worker_stage2(args,log_s1_name,log_s2_name) # Stage 2

def main_worker_stage1(args,log_s1_name):
    logs_dir_root = osp.join(args.logs_dir + '/' + log_s1_name)
    trial = args.trial

    global cls_token_num
    cls_token_num=args.cls_token_num
    start_epoch=0
    best_mAP=0
    best_epoch = 0
    data_dir = args.data_dir
    args.logs_dir = osp.join(logs_dir_root, str(trial))
    start_time = time.monotonic()

    cudnn.benchmark = True  # 启用 PyTorch 中的 cuDNN 基准测试模式，以加速卷积操作。
    # cudnn.benchmark = True
    sys.stdout = Logger(osp.join(args.logs_dir, str(trial) + 'log.txt'))

    print("==========\nArgs:{}\n==========".format(args))
    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset_ir = get_data('sysu_ir', args.data_dir)
    dataset_rgb = get_data('sysu_rgb', args.data_dir)

    # Create model
    model = models.create(args.arch,img_size=(args.height,args.width),drop_path_rate=args.drop_path_rate0
            , pretrained_path = args.pretrained_path,hw_ratio=args.hw_ratio, conv_stem=args.conv_stem,cls_token_num=cls_token_num)
    model.cuda()
    model = nn.DataParallel(model)
    
    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr0, momentum=0.9, weight_decay=args.weight_decay0) 
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    # Trainer
    trainer = ClusterContrastTrainer_DCL(model)

    
    for epoch in range(args.epochs):
        with torch.no_grad():
            if epoch == 0:
                # DBSCAN cluster
                ir_eps = 0.6
                print('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
                cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                rgb_eps = 0.6
                print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
                cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)

            print('==> Create pseudo labels for unlabeled RGB data')

            cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                             64, args.workers,
                                             testset=sorted(dataset_rgb.train))
            features_rgb,_, ture_ids_rgb = extract_features(model, cluster_loader_rgb, print_freq=50,mode=1)
            del cluster_loader_rgb
            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)
            features_rgb_norm=F.normalize(features_rgb, dim=1)
            ture_ids_rgb = torch.cat([ture_ids_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)

            print('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                             64, args.workers,
                                             testset=sorted(dataset_ir.train))
            features_ir, _,ture_ids_ir = extract_features(model, cluster_loader_ir, print_freq=50,mode=2)
            del cluster_loader_ir
            features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)
            features_ir_norm =F.normalize(features_ir, dim=1)
            ture_ids_ir = torch.cat([ture_ids_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)

            rerank_dist_ir = compute_jaccard_distance(features_ir_norm , k1=args.k1, k2=args.k2,search_option=3)#rerank_dist_all_jacard[features_rgb.size(0):,features_rgb.size(0):]#
            pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
            rerank_dist_rgb = compute_jaccard_distance(features_rgb_norm , k1=args.k1, k2=args.k2,search_option=3)#rerank_dist_all_jacard[:features_rgb.size(0),:features_rgb.size(0)]#
            pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)
            del rerank_dist_rgb
            del rerank_dist_ir
            del features_rgb_norm
            del features_ir_norm
            num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
            num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)


            pseudo_labels_rgb1 = torch.from_numpy(pseudo_labels_rgb)
            pseudo_labels_ir1 = torch.from_numpy(pseudo_labels_ir)
            print_intra_acc(ture_ids_rgb, ture_ids_ir, pseudo_labels_rgb1, pseudo_labels_ir1)

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])
            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir)
        cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)


        # memory_ir = ClusterMemory(model.module.num_features*cls_token_num, num_cluster_ir, temp=args.temp,
        #                        momentum=args.momentum0, use_hard=args.use_hard).cuda()
        # memory_rgb = ClusterMemory(model.module.num_features*cls_token_num, num_cluster_rgb, temp=args.temp,
        #                        momentum=args.momentum0, use_hard=args.use_hard).cuda()
        # memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
        # memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()
        # del cluster_features_rgb, cluster_features_ir
        #
        # trainer.memory_ir = memory_ir
        # trainer.memory_rgb = memory_rgb


        memory_ir = ClusterMemory(768 * cls_token_num, num_cluster_ir, temp=args.temp, momentum=args.momentum,
                                  mode=args.memorybank, smooth=args.smooth,
                                      num_instances=args.num_instances).cuda()

        memory_rgb = ClusterMemory(768 * cls_token_num, num_cluster_rgb, temp=args.temp, momentum=args.momentum,
                                  mode=args.memorybank, smooth=args.smooth,
                                      num_instances=args.num_instances).cuda()

        if args.memorybank == 'CM':
            memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
            memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()

        elif args.memorybank == 'CMhcl':
            memory_ir.features = F.normalize(cluster_features_ir.repeat(2, 1), dim=1).cuda()
            memory_rgb.features = F.normalize(cluster_features_rgb.repeat(2, 1), dim=1).cuda()

        trainer.memory_ir = memory_ir
        trainer.memory_rgb= memory_rgb




        pseudo_labeled_dataset_ir = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
        print('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))
        
        pseudo_labeled_dataset_rgb = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
        print('==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))

        # ########################
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        height = args.height
        width = args.width
        train_transformer_rgb = T.Compose([
            T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            # T.RandomGrayscale(p=0.1),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability=0.5)
        ])

        train_transformer_rgb1 = T.Compose([
            T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability=0.5),
            ChannelExchange(gray=2),
        ])

        transform_thermal = T.Compose([
            T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability=0.5),
            ChannelAdapGray(probability=0.5)])


        train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal)

        train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                        (args.batch_size//2), args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)

        train_loader_ir.new_epoch()
        train_loader_rgb.new_epoch()

        trainer.train(epoch, train_loader_ir,train_loader_rgb, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader_ir))

        if epoch >= 0:
##############################
            args.test_batch=64
            args.img_w=args.width
            args.img_h=args.height
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            transform_test = T.Compose([
                T.ToPILImage(),
                T.Resize((args.img_h,args.img_w)),
                T.ToTensor(),
                normalize,
            ])
            mode='all'
            data_path = data_dir
            query_img, query_label, query_cam = process_query_sysu(data_path, mode=mode)
            nquery = len(query_label)
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
            query_feat_fc = extract_query_feat(model,query_loader,nquery) # mode=2
            for trial in range(1):
                gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=mode, trial=trial)
                ngall = len(gall_label)
                trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

                gall_feat_fc = extract_gall_feat(model,trial_gall_loader,ngall) # mode=1

                # fc feature
                distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
                cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)

                if trial == 0:
                    all_cmc = cmc
                    all_mAP = mAP
                    all_mINP = mINP

                else:
                    all_cmc = all_cmc + cmc
                    all_mAP = all_mAP + mAP
                    all_mINP = all_mINP + mINP

                print('Test Trial: {}'.format(trial))
                print(
                    'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

            cmc = all_cmc
            mAP = all_mAP
            mINP = all_mINP
            print('All Average:')
            print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
            #################################
            is_best = (mAP > best_mAP)
            if is_best:
                best_R1 = cmc[0]
                best_mAP = max(mAP, best_mAP)
                best_epoch = epoch
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print(
                '\n * Finished epoch {:3d}   model R1: {:5.1%}  model mAP: {:5.1%}   best R1: {:5.1%}   best mAP: {:5.1%}(best_epoch:{})\n'.
                format(epoch, cmc[0], mAP, best_R1, best_mAP, best_epoch))
            ############################
        lr_scheduler.step()
        print("the learning rate is ", optimizer.state_dict()['param_groups'][0]['lr'])
        print('---------------------------------------------------------------------')


def main_worker_stage2(args,log_s1_name,log_s2_name):
    logs_dir_root = osp.join('logs/' + log_s2_name)
    trial = args.trial
    global cls_token_num
    cls_token_num = args.cls_token_num
    start_epoch = 0
    best_mAP = 0
    best_R1 = 0
    best_epoch = 0
    data_dir = args.data_dir
    args.logs_dir = osp.join(logs_dir_root, str(trial))
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, str(trial) + 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))


    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset_ir = get_data('sysu_ir', args.data_dir, trial=trial)
    dataset_rgb = get_data('sysu_rgb', args.data_dir, trial=trial)

    # Create model

    model = models.create(args.arch, img_size=(args.height, args.width), drop_path_rate=args.drop_path_rate1
                          , pretrained_path=args.pretrained_path, hw_ratio=args.hw_ratio, conv_stem=args.conv_stem,
                          cls_token_num=cls_token_num)
    model.cuda()
    model = nn.DataParallel(model)  # ,output_device=1)

    checkpoint = load_checkpoint(osp.join('logs/' + log_s1_name + '/' + str(trial)+ '/' + 'model_best.pth.tar'))
    # checkpoint = load_checkpoint(osp.join('./logs/' + 'sysu_s2' + '/' + str(trial), 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])


    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    # optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(params, lr=args.lr1, momentum=0.9, weight_decay=args.weight_decay1) ##########args.lr1

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 50, eta_min=1e-7)  # 1e-7

    # Trainer
    trainer = ClusterContrastTrainer(model)   ####################
    for epoch in range(args.epochs):

        
        with torch.no_grad():
            if epoch == 0:
                # DBSCAN cluster
                ir_eps = 0.6  #0.6
                print('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
                cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                rgb_eps = 0.6
                print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
                cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)

            print('==> Create pseudo labels for unlabeled RGB data')
            cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                             64, args.workers,
                                             testset=sorted(dataset_rgb.train))
            features_rgb, features_rgb_pa, ture_ids_rgb = extract_features(model, cluster_loader_rgb, print_freq=50,mode=1)
            del cluster_loader_rgb,
            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)  #############
            features_rgb_norm=F.normalize(features_rgb, dim=1)
            features_rgb_pa = torch.cat([features_rgb_pa[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)
            ture_ids_rgb = torch.cat([ture_ids_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)
            
            print('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                             64, args.workers,
                                             testset=sorted(dataset_ir.train))
            features_ir, features_ir_pa, ture_ids_ir = extract_features(model, cluster_loader_ir, print_freq=50,mode=2)
            del cluster_loader_ir
            features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)
            features_ir_norm=F.normalize(features_ir, dim=1)
            features_ir_pa = torch.cat([features_ir_pa[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)
            ture_ids_ir = torch.cat([ture_ids_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)

            rerank_dist_ir = compute_jaccard_distance(features_ir_norm, k1=30, k2=args.k2,search_option=3) #args.k1
            pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
            rerank_dist_rgb = compute_jaccard_distance(features_rgb_norm, k1=30, k2=args.k2,search_option=3)
            pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)
            
            del rerank_dist_rgb
            del rerank_dist_ir
            del features_rgb_norm
            del features_ir_norm

            num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
            num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)

            pseudo_labels_rgb1 = torch.from_numpy(pseudo_labels_rgb)
            pseudo_labels_ir1 = torch.from_numpy(pseudo_labels_ir)
            print_intra_acc(ture_ids_rgb, ture_ids_ir, pseudo_labels_rgb1, pseudo_labels_ir1)

        @torch.no_grad()
        def generate_cluster_features(labels, features, features_pa):
            centers = collections.defaultdict(list)
            centers_pa = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])
                centers_pa[labels[i]].append(features_pa[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers_pa = [
                torch.stack(centers_pa[idx], dim=0).mean(dim=0) for idx in sorted(centers_pa.keys())
            ]

            centers = torch.stack(centers, dim=0)
            centers_pa = torch.stack(centers_pa, dim=0)
            return centers, centers_pa



        @torch.no_grad()
        def generate_cluster_features_corr(labels, features):
            centers = collections.defaultdict(list)
            prototypes = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            for i, feat in centers.items():
                length = len(feat)
                feat = torch.stack(feat, dim=0)
                normalized_feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)
                S = cosine_similarity(normalized_feat)
                rho_list = []
                for j in range(length):
                    rho = np.sign(S[j] - 0.5).sum()
                    rho_list.append(rho)
                top_k_indices = heapq.nlargest(20, range(len(rho_list)), key=rho_list.__getitem__)

                for j in range(len(top_k_indices)):
                    prototypes[i].append(feat[top_k_indices[j]])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]
            centers = torch.stack(centers, dim=0)


            centers_proto = [
                torch.stack(prototypes[idx], dim=0).mean(0) for idx in sorted(prototypes.keys())
            ]
            centers_proto = torch.stack(centers_proto, dim=0)
            return centers, centers_proto

        def correct_label(features, pseudo_labels, prototypes):
            pseudo_labels_hat = copy.deepcopy(pseudo_labels)

            for i in range(len(features)):
                labels = pseudo_labels[i]
                if labels == -1:
                    continue
                s = (np.dot(features[i], prototypes.T) / (
                            np.linalg.norm(features[i]) * np.linalg.norm(prototypes, axis=1)))
                c_label = np.argmax(s)
                pseudo_labels_hat[i] = c_label

            return pseudo_labels_hat

        # cluster_features_ir, prototypes_ir = generate_cluster_features_corr(pseudo_labels_ir, features_ir)
        cluster_features_rgb, prototypes_rgb = generate_cluster_features_corr(pseudo_labels_rgb, features_rgb)
        print("Correcting label")
        # pseudo_labels_ir_hat = correct_label(features_ir, pseudo_labels_ir, prototypes_ir)
        pseudo_labels_ir_hat = pseudo_labels_ir
        pseudo_labels_rgb_hat = correct_label(features_rgb, pseudo_labels_rgb, prototypes_rgb)


        # pseudo_labels_ir_hat = pseudo_labels_ir
        # # pseudo_labels_rgb_hat = correct_label(features_rgb, pseudo_labels_rgb, prototypes_rgb)
        # pseudo_labels_rgb_hat = pseudo_labels_rgb
        print("Correcting label finished")



        pseudo_labels_rgb2 = torch.from_numpy(pseudo_labels_rgb_hat)
        pseudo_labels_ir2 = torch.from_numpy(pseudo_labels_ir_hat)


        print_intra_acc(ture_ids_rgb, ture_ids_ir, pseudo_labels_rgb2, pseudo_labels_ir2)

        cluster_features_ir, cluster_features_ir_pa = generate_cluster_features(pseudo_labels_ir_hat, features_ir,
                                                                                features_ir_pa)
        cluster_features_rgb, cluster_features_rgb_pa = generate_cluster_features(pseudo_labels_rgb_hat, features_rgb,
                                                                                  features_rgb_pa)



        def update_label(features, pseudo_labels, centers):
            pseudo_labels_hat = copy.deepcopy(pseudo_labels)  # 创建伪标签的深拷贝，避免修改原始标签。
            for i in range(len(features)):
                labels = pseudo_labels[i]
                if labels == -1:
                    # 计算当前样本与所有原型（新的中心）的余弦相似度：
                    s = (np.dot(features[i], centers.T) / (
                            np.linalg.norm(features[i]) * np.linalg.norm(centers, axis=1)))
                    c_label = np.argmax(s)  # 找到相似度最高的原型索引
                    pseudo_labels_hat[i] = c_label  # 更新label
            return pseudo_labels_hat



        pseudo_labels_ir_update = update_label(features_ir, pseudo_labels_ir2, cluster_features_ir)
        pseudo_labels_rgb_update = update_label(features_rgb, pseudo_labels_rgb2, cluster_features_rgb)



        memory_ir = ClusterMemory(768 * cls_token_num, num_cluster_ir, temp=args.temp, momentum=args.momentum,
                                  mode=args.memorybank, smooth=args.smooth,
                                      num_instances=args.num_instances).cuda()

        memory_rgb = ClusterMemory(768 * cls_token_num, num_cluster_rgb, temp=args.temp, momentum=args.momentum,
                                  mode=args.memorybank, smooth=args.smooth,
                                      num_instances=args.num_instances).cuda()

        if args.memorybank == 'CM':
            memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
            memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()

        elif args.memorybank == 'CMhcl':
            memory_ir.features = F.normalize(cluster_features_ir.repeat(2, 1), dim=1).cuda()
            memory_rgb.features = F.normalize(cluster_features_rgb.repeat(2, 1), dim=1).cuda()

        trainer.memory_ir = memory_ir
        trainer.memory_rgb= memory_rgb

        pseudo_labeled_dataset_ir = []

        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir_hat)):
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))

        print('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))

        pseudo_labeled_dataset_rgb = []

        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb_hat)):
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))

        print('==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))


        def shortest_dist(x, y):
            #############################
            #############################
            #############################
            #############################
            #############################
            #############################
            #############################
            #############################
            return #####

        def compute_distance_matrix(X, Y):
            #############################
            #############################
            #############################
            #############################
            #############################
            #############################
            #############################
            #############################
            return #####


        def min_max_normalize(d: dict) -> dict:
            #############################
            #############################
            #############################
            #############################
            #############################
            #############################
            #############################
            #############################

            return ####


        X = cluster_features_rgb_pa
        Y = cluster_features_ir_pa

        distance_matrix = compute_distance_matrix(X, Y)

        # distance_matrix = torch.cdist(cluster_features_rgb,cluster_features_ir)

        ####################### OTPM
        print("Optimal Transport Pseudo-label Matching")
        cost = distance_matrix
        a = np.ones(cost.shape[0]) / cost.shape[0]
        b = np.ones(cost.shape[1]) / cost.shape[1]
        M = np.array(cost)
        result = ot.sinkhorn(a, b, M, reg=1, numItermax=5000, stopThr=1e-5)
        assign1 = np.argmax(result, axis=1)
        assign2 = np.argmax(result, axis=0)
        i2r = {}
        r2i = {}
        for i in range(num_cluster_rgb):
            r2i[i] = assign1[i]
        for i in range(num_cluster_ir):
            i2r[i] = assign2[i]
        print("Optimal Transport Pseudo-label Matching Done")


        assign1_p = np.max(result, axis=1)
        assign2_p = np.max(result, axis=0)
        r2i_p = {}
        i2r_p = {}
        for i in range(num_cluster_rgb):
            r2i_p[i] = assign1_p[i]
        for i in range(num_cluster_ir):
            i2r_p[i] = assign2_p[i]

        r2i_p = min_max_normalize(r2i_p)
        i2r_p = min_max_normalize(i2r_p)


        pseudo_labels_rgb_list = pseudo_labels_rgb_update.tolist()
        pseudo_labels_ir_list = pseudo_labels_ir_update.tolist()
        IR_instance_RGB_label = conversion_(pseudo_labels_ir_list, i2r)
        RGB_instance_IR_label = conversion_(pseudo_labels_rgb_list, r2i)

        RGB_instance_IR_label = torch.tensor(RGB_instance_IR_label)
        IR_instance_RGB_label = torch.tensor(IR_instance_RGB_label)

        print_cm_acc(ture_ids_rgb, ture_ids_ir, pseudo_labels_rgb_update, pseudo_labels_ir_update,
                     RGB_instance_IR_label, IR_instance_RGB_label)




        del cluster_features_ir, cluster_features_rgb, cluster_features_ir_pa,cluster_features_rgb_pa




        color_aug = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)#T.
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        height=args.height
        width=args.width
        train_transformer_rgb = T.Compose([
            color_aug,
            T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            # T.RandomGrayscale(p=0.1),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability = 0.5)
        ])
        
        train_transformer_rgb1 = T.Compose([
            color_aug,
            T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            #T.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability = 0.5),
            ChannelExchange(gray = 2),
        ])

        transform_thermal = T.Compose( [
            color_aug,
            T.Resize((height, width), interpolation=InterpolationMode.BICUBIC),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability = 0.5),
            ChannelAdapGray(probability =0.5)])

        train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,  # args.batch_size
                                        trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,train_transformer=transform_thermal)

        train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                        (args.batch_size//2), args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,train_transformer=train_transformer_rgb,train_transformer1=train_transformer_rgb1)

        train_loader_ir.new_epoch()
        train_loader_rgb.new_epoch()
        

        trainer.train(epoch, train_loader_ir,train_loader_rgb, optimizer,
                print_freq=args.print_freq, train_iters=len(train_loader_ir), i2r=i2r, r2i=r2i,
                i2r_p=i2r_p, r2i_p=r2i_p)

        del train_loader_ir, train_loader_rgb

        if epoch >= 0:
            args.test_batch = 64
            args.img_w=args.width
            args.img_h=args.height
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            transform_test = T.Compose([
                T.ToPILImage(),
                T.Resize((args.img_h,args.img_w)),
                T.ToTensor(),
                normalize,
            ])
            mode='all'
            data_path = data_dir
            query_img, query_label, query_cam = process_query_sysu(data_path, mode=mode)
            nquery = len(query_label)
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
            query_feat_fc = extract_query_feat(model,query_loader,nquery)
            for trial in range(1):
                gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=mode, trial=trial)
                ngall = len(gall_label)
                trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
                trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

                gall_feat_fc = extract_gall_feat(model,trial_gall_loader,ngall)

                # fc feature
                distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
                cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)

                if trial == 0:
                    all_cmc = cmc
                    all_mAP = mAP
                    all_mINP = mINP

                else:
                    all_cmc = all_cmc + cmc
                    all_mAP = all_mAP + mAP
                    all_mINP = all_mINP + mINP

                print('Test Trial: {}'.format(trial))
                print(
                    'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

            cmc = all_cmc
            mAP = all_mAP
            mINP = all_mINP
            print('All Average:')
            print('FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

            #################################
            is_best = (cmc[0] > best_R1)
            if is_best:
                best_R1 = max(cmc[0], best_R1)
                best_mAP = mAP
                best_epoch = epoch
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print(
                '\n * Finished epoch {:3d}   model R1: {:5.1%}  model mAP: {:5.1%}   best R1: {:5.1%}   best mAP: {:5.1%}(best_epoch:{})\n'.
                format(epoch, cmc[0], mAP, best_R1, best_mAP, best_epoch))
            ############################
        lr_scheduler.step()

        print("the learning rate is ", optimizer.state_dict()['param_groups'][0]['lr'])
        print('---------------------------------------------------------------------')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Augmented Dual-Contrastive Aggregation Learning for USL-VI-ReID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='sysu')
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('--test-batch', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=0)
    parser.add_argument('--height', type=int, default=384, help="input height")  # 288   384
    parser.add_argument('--width', type=int, default=128, help="input width")    # 144   128
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='vit_base',
                        choices=models.names())
    parser.add_argument('-pp', '--pretrained-path', type=str, default='vit_base_ics_cfs_lup.pth')
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for the hybrid memory")

    parser.add_argument('-mb', '--memorybank', type=str, default='CMhcl',
                        choices=['CM', 'CMhcl'])
    parser.add_argument('--smooth', type=float, default=0, help="label smoothing")

    #vit
    parser.add_argument('--drop-path-rate0', type=float, default=0.3)
    parser.add_argument('--drop-path-rate1', type=float, default=0.3)
    parser.add_argument('--hw-ratio', type=int, default=2)
    parser.add_argument('--self-norm', action="store_true")
    parser.add_argument('--conv-stem', action="store_true")
    # optimizer
    parser.add_argument('--lr0', type=float, default=0.00035,
                        help="learning rate0")
    parser.add_argument('--lr1', type=float, default=0.000035,
                        help="learning rate1")
    parser.add_argument('--weight-decay0', type=float, default=5e-4)  #5e-4
    parser.add_argument('--weight-decay1', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--step-size', type=int, default=10)
    parser.add_argument('--trial', type=int, default=1)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=20)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default="data/SYSU-MM01/")
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")  
    parser.add_argument('--no-cam',  action="store_true")

    parser.add_argument('--cls-token-num', type=int, default=4)

    main()
