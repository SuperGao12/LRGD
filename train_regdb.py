# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import math
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN
from PIL import Image
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from evel import *
from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.trainers import ClusterContrastTrainer_DCL, ClusterContrastTrainer
from clustercontrast.evaluators import Evaluator, extract_features
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor, Preprocessor_aug
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam
import os
import torch.utils.data as data
from torch.autograd import Variable
from ChannelAug import ChannelAdap, ChannelAdapGray, ChannelRandomErasing, ChannelExchange, Gray
from sklearn.metrics.pairwise import cosine_similarity
import heapq
import copy
import ot
import warnings
from collections import defaultdict
warnings.filterwarnings("ignore")


def get_data(name, data_dir, trial=0):
    dataset = datasets.create(name, root=data_dir, trial=trial)
    return dataset

class channel_select(object):
    def __init__(self, channel=0):
        self.channel = channel

    def __call__(self, img):
        if self.channel == 3:
            img_gray = img.convert('L')
            np_img = np.array(img_gray, dtype=np.uint8)
            img_aug = np.dstack([np_img, np_img, np_img])
            img_PIL = Image.fromarray(img_aug, 'RGB')
        else:
            np_img = np.array(img, dtype=np.uint8)
            np_img = np_img[:, :, self.channel]
            img_aug = np.dstack([np_img, np_img, np_img])
            img_PIL = Image.fromarray(img_aug, 'RGB')
        return img_PIL


def get_train_loader_ir(args, dataset, height, width, batch_size, workers,
                        num_instances, iters, trainset=None, no_cam=False, train_transformer=None,
                        train_transformer1=None):
    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    # 如果 trainset 为 None，则使用 dataset.train 作为训练集，并对其进行排序；否则，使用传入的 trainset 并排序。
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        # 用于判断是否使用随机多重画廊采样策略（Random Multiple Gallery Sampler）。如果 num_instances 大于 0，则使用该策略。
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
            # 如果 no_cam 为 True，则使用不考虑摄像头信息的采样器 RandomMultipleGallerySamplerNoCam；
            # 否则，使用普通的采样器 RandomMultipleGallerySampler。
    else:
        sampler = None
    if train_transformer1 is None:
        train_loader = IterLoader(
            DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
        # train_transformer1 为 None，则使用 Preprocessor 类进行数据预处理，该类只使用 train_transformer 进行图像变换
    else:
        train_loader = IterLoader(
            DataLoader(Preprocessor_aug(train_set, root=dataset.images_dir, transform=train_transformer,
                                        transform1=train_transformer1),
                       batch_size=batch_size, num_workers=workers, sampler=sampler,
                       shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
        # 如果 train_transformer1 不为 None，则使用 Preprocessor_aug 类进行数据预处理，
        # 该类使用 train_transformer 和 train_transformer1 进行图像变换。

    # 使用 DataLoader 类创建一个普通的数据加载器，配置批次大小、工作进程数量、采样器、是否打乱数据、
    # 是否使用锁页内存以及是否丢弃最后一个不完整的批次。
    # 使用 IterLoader 类将普通数据加载器包装成迭代式数据加载器，并指定迭代次数为 iters
    return train_loader
    # 该函数根据输入的参数配置数据加载器，支持不同的采样策略和数据预处理方式，
    # 最终返回一个用于训练的迭代式数据加载器，方便在训练过程中按指定的迭代次数加载数据。


def get_train_loader_color(args, dataset, height, width, batch_size, workers,
                           num_instances, iters, trainset=None, no_cam=False, train_transformer=None,
                           train_transformer1=None):
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


def get_test_loader(dataset, height, width, batch_size, workers, testset=None, test_transformer=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    # T.Normalize 是 torchvision.transforms 中的一个类，用于对图像数据进行归一化处理。
    # 这里使用的均值和标准差是 ImageNet 数据集的统计值，是一种常见的图像归一化设置。
    if test_transformer is None:
        test_transformer = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.ToTensor(),
            normalizer
        ])

    if testset is None:
        # 如果 testset 参数未提供，则将数据集的查询集和画廊集合并为一个集合，并转换为列表。
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)
    # Preprocessor 返回处理后的图像、文件名、标签、摄像头ID和索引

    return test_loader


def create_model(args):
    if args.arch == 'vit_base':
        print('vit_base')
        model = models.create(args.arch, img_size=(args.height, args.width), drop_path_rate=args.drop_path_rate
                              , pretrained_path=args.pretrained_path, hw_ratio=args.hw_ratio, conv_stem=args.conv_stem)
    else:
        model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                              num_classes=0, pooling_type=args.pooling_type)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)  # ,output_device=1)
    return model


# 该类的主要功能是封装测试数据，包括读取测试图像、调整图像大小、存储图像和标签信息，
# 并提供索引访问和长度查询的功能，方便后续使用 DataLoader 进行批量数据加载。
class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size=(128, 384)):
        test_image = []
        for i in range(len(test_img_file)):
            # 遍历 test_img_file 列表中的每个图像文件路径
            img = Image.open(test_img_file[i])
            # 使用 Image.open 方法打开图像文件
            img = img.resize((img_size[0], img_size[1]), Image.LANCZOS)
            pix_array = np.array(img)
            test_image.append(pix_array)
            # 调整大小后的图像转换为 NumPy 数组，并添加到 test_image 列表中。
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform
        # 将 test_image 列表转换为 NumPy 数组，并存储在类的属性 self.test_image 中。
        # 将传入的 test_label 存储在类的属性 self.test_label 中。
        # 将传入的 transform 存储在类的属性 self.transform 中，用于后续的图像预处理

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    # 获取样本：根据索引 index 从 self.test_image 和 self.test_label 中获取对应的图像和标签。
    # 图像预处理：如果 self.transform 不为 None，则对图像 img1 应用该预处理变换。
    # 返回结果：返回处理后的图像和对应的标签

    def __len__(self):
        return len(self.test_image)


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_gall_feat(model, gall_loader, ngall):
    # pool_dim = 2048
    pool_dim = 768 * cls_token_num
    net = model
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc,_ = net(input, input, 2)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1,_ = net(flip_input, flip_input, 2)
            feature_fc = (feat_fc.detach() + feat_fc_1.detach()) / 2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            gall_feat_fc[ptr:ptr + batch_num, :] = feature_fc.cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return gall_feat_fc


def extract_query_feat(model, query_loader, nquery):

    # pool_dim = 2048
    pool_dim = 768 * cls_token_num
    net = model
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    # query_feat_pool 和 query_feat_fc：分别用于存储池化层和全连接层的查询特征，初始化为全零数组，形状为 (nquery, pool_dim)。
    with torch.no_grad():
        # 上下文管理器，用于关闭梯度计算，因为在特征提取阶段不需要进行反向传播，这样可以节省内存和计算资源。
        for batch_idx, (input, label) in enumerate(query_loader):
            # 遍历 query_loader 中的每个批次数据：
            batch_num = input.size(0)
            # batch_num：获取当前批次的样本数量。
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            # 将输入数据和翻转后的数据移动到 GPU 上，并包装为 Variable（在较新的 PyTorch 版本中，Variable 已被弃用，可直接使用张量）。
            feat_fc,_ = net(input, input, 1)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1,_ = net(flip_input, flip_input, 1)
            # feat_fc = net(input, input, 1) 和 feat_fc_1 = net(flip_input, flip_input, 1)：分别将原始输入和翻转后的输入传入模型，得到全连接层的特征。
            feature_fc = (feat_fc.detach() + feat_fc_1.detach()) / 2
            # 将原始输入和翻转输入得到的特征求平均，以增强特征的稳定性。detach() 方法用于将张量从计算图中分离，避免梯度传播。
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            # 计算特征的 L2 范数。
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            # 对特征进行 L2 归一化，使特征向量的模长为 1。
            query_feat_fc[ptr:ptr + batch_num, :] = feature_fc.cpu().numpy()
            # 将处理后的特征从 GPU 移动到 CPU 并转换为 NumPy 数组，存储到 query_feat_fc 中相应的位置。

            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return query_feat_fc


def process_test_regdb(img_dir, trial=1, modal='visible'):
    # 函数的主要功能是处理 RegDB 数据集的测试数据，根据指定的模态（可见光或热红外）和试验编号，
    # 从对应的文本文件中读取测试数据的图像路径和标签信息，并将其返回。
    if modal == 'visible':
        input_data_path = img_dir + 'idx/test_visible_{}'.format(trial) + '.txt'
    elif modal == 'thermal':
        input_data_path = img_dir + 'idx/test_thermal_{}'.format(trial) + '.txt'
        # 如果 modal 为 'visible'，则构建可见光模态的测试数据文件路径，文件名为 test_visible_<trial>.txt。
        # 如果 modal 为 'thermal'，则构建热红外模态的测试数据文件路径，文件名为 test_thermal_<trial>.txt。

    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        # 图像路径提取：使用列表推导式遍历 data_file_list 中的每一行，通过 split(' ') 方法将每行按空格分割成两部分，取第一部分作为图像文件名，再与 img_dir 拼接成完整的图像路径，存储在 file_image 列表中。
        # 标签提取：同样使用列表推导式，取每行分割后的第二部分，并将其转换为整数类型，作为图像的标签，存储在 file_label 列表中

    return file_image, np.array(file_label)


def eval_regdb(distmat, q_pids, g_pids, max_rank=20):
    # # 此函数用于评估在 RegDB 数据集上的行人重识别性能，接收距离矩阵、查询集的行人 ID 和画廊集的行人 ID 作为输入
    #     # distmat: 形状为 (num_q, num_g) 的距离矩阵，其中 num_q 是查询样本的数量，num_g 是画廊样本的数量
    #     # q_pids: 查询样本的行人 ID 数组
    #     # g_pids: 画廊样本的行人 ID 数组
    #     # max_rank: 计算 CMC（Cumulative Matching Characteristic）曲线时考虑的最大排名，默认为 20
    num_q, num_g = distmat.shape
    #  num_q, num_g = distmat.shape
    #  获取距离矩阵的行数（查询样本数量）和列数（画廊样本数量）
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
        # # 如果gallery样本数量小于 max_rank，则将 max_rank 更新为画廊样本数量，并打印提示信息
    indices = np.argsort(distmat, axis=1)
    # 对距离矩阵的每一行进行排序，返回排序后的索引。这样，indices[i] 表示第 i 个查询样本与所有画廊样本的距离从小到大排序后的索引
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # g_pids[indices] 根据排序后的索引获取画廊样本的行人 ID，q_pids[:, np.newaxis] 将查询样本的行人 ID 扩展为二维数组
    # 比较两者是否相等，得到一个布尔矩阵，将其转换为整数类型（0 或 1），表示匹配结果

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query
    # # 初始化列表用于存储每个查询样本的 CMC 曲线、平均精度（AP）和平均归一化精度（INP）
    #     # num_valid_q 用于记录有效的查询样本数量

    # only two cameras
    q_camids = np.ones(num_q).astype(np.int32)
    g_camids = 2 * np.ones(num_g).astype(np.int32)
    # # 假设查询集和画廊集分别来自两个不同的摄像头，将查询集的摄像头 ID 设为 1，画廊集的摄像头 ID 设为 2

    for q_idx in range(num_q):
        # # 遍历每个查询样本
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        # # 获取当前查询样本的行人 ID 和摄像头 ID

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        # order 是第 q_idx 个查询样本与所有画廊样本的距离从小到大排序后的索引
        # remove 是一个布尔数组，标记那些与当前查询样本具有相同行人 ID 和摄像头 ID 的画廊样本
        # keep 是 remove 的取反，标记那些需要保留的画廊样本

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue
        # raw_cmc 是当前查询样本与保留的画廊样本的匹配结果，是一个二进制向量
        # 如果 raw_cmc 中没有任何值为 1 的元素，说明当前查询样本的行人 ID 不在画廊集中，跳过该查询样

        cmc = raw_cmc.cumsum()
        # 对 raw_cmc 进行累积求和，得到 CMC 曲线

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(raw_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)
        all_INP.append(inp)
        # 计算平均归一化精度（INP）
        # pos_idx 是 raw_cmc 中值为 1 的元素的索引
        # pos_max_idx 是这些索引中的最大值
        # inp 是 CMC 曲线在 pos_max_idx 处的值除以 (pos_max_idx + 1)
        # 将计算得到的 INP 添加到 all_INP 列表中

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return all_cmc, mAP, mINP


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    log_s1_name = 'regdb_s1'
    log_s2_name = 'regdb_s2'
    # log_s1_name: 第一阶段的日志名称。
    # log_s2_name: 第二阶段的日志名称。
    # 这些名称可能用于保存训练日志、模型权重或其他相关信息。
    # main_worker_stage1(args, log_s1_name)  # 第一阶段的主工作函数，
    main_worker_stage2(args, log_s1_name, log_s2_name)  # 第二阶段的主工作函数


def main_worker_stage1(args, log_s1_name):
    logs_dir_root = osp.join(args.logs_dir + '/' + log_s1_name)
    trial = args.trial

    global cls_token_num
    cls_token_num = args.cls_token_num

    # global start_epoch, best_mAP
    start_epoch = 0
    best_mAP = 0
    best_R1 = 0
    best_epoch = 0
    data_dir = args.data_dir
    args.logs_dir = osp.join(logs_dir_root, str(trial))
    start_time = time.monotonic()  # 函数记录程序的启动时间，以便后续计算程序运行的总时间。

    cudnn.benchmark = True  # 启用 PyTorch 中的 cuDNN 基准测试模式，以加速卷积操作。

    sys.stdout = Logger(osp.join(args.logs_dir, str(trial) + 'log.txt'))  # 用于将输出同时写入到文件和控制台。
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset_ir = get_data('regdb_ir', args.data_dir, trial=trial)
    dataset_rgb = get_data('regdb_rgb', args.data_dir, trial=trial)


    # Create model
    model = models.create(args.arch, img_size=(args.height, args.width), drop_path_rate=args.drop_path_rate0
                          , pretrained_path=args.pretrained_path, hw_ratio=args.hw_ratio, conv_stem=args.conv_stem,
                          cls_token_num=cls_token_num)
    model.cuda()
    model = nn.DataParallel(model)  # ,output_device=1)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]

    # optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)  # 使用筛选出的参数初始化一个 Adam 优化器。
    optimizer = torch.optim.SGD(params, lr=args.lr0, momentum=0.9, weight_decay=args.weight_decay0)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size,gamma=0.1)  # 为优化器配置一个步长衰减的学习率调度器。
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)


    # Trainer
    trainer = ClusterContrastTrainer_DCL(model)

    for epoch in range(args.epochs):
        with torch.no_grad():
            if epoch == 0:
                # DBSCAN cluster
                ir_eps = 0.3

                print('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
                cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)

                rgb_eps = 0.3
                print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
                cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)

            print('==> Create pseudo labels for unlabeled RGB data')

            cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                                 64, args.workers,
                                                 testset=sorted(dataset_rgb.train))

            features_rgb, _,ture_ids_rgb = extract_features(model, cluster_loader_rgb, print_freq=50, mode=1)

            del cluster_loader_rgb,
            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)
            features_rgb_norm = F.normalize(features_rgb, dim=1)
            ture_ids_rgb = torch.cat([ture_ids_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)

            print('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                                64, args.workers,
                                                testset=sorted(dataset_ir.train))
            features_ir, _,ture_ids_ir = extract_features(model, cluster_loader_ir, print_freq=50, mode=2)
            del cluster_loader_ir
            features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)
            features_ir_norm = F.normalize(features_ir, dim=1)
            ture_ids_ir = torch.cat([ture_ids_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)


            rerank_dist_ir = compute_jaccard_distance(features_ir_norm, k1=args.k1, k2=args.k2, search_option=3)
            pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
            rerank_dist_rgb = compute_jaccard_distance(features_rgb_norm, k1=args.k1, k2=args.k2, search_option=3)
            pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)

            del rerank_dist_rgb
            del rerank_dist_ir
            del features_ir_norm
            del features_rgb_norm
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

        memory_ir = ClusterMemory(model.module.num_features*cls_token_num, num_cluster_ir, temp=args.temp,
                                  momentum=args.momentum, mode=args.memorybank, smooth=args.smooth,
                                  num_instances=args.num_instances).cuda()

        memory_rgb = ClusterMemory(model.module.num_features*cls_token_num, num_cluster_rgb, temp=args.temp,
                                   momentum=args.momentum, mode=args.memorybank, smooth=args.smooth,
                                   num_instances=args.num_instances).cuda()

        if args.memorybank == 'CM':
            memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
            memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()
        elif args.memorybank == 'CMhcl':
            memory_ir.features = F.normalize(cluster_features_ir.repeat(2, 1), dim=1).cuda()
            memory_rgb.features = F.normalize(cluster_features_rgb.repeat(2, 1), dim=1).cuda()

        trainer.memory_ir = memory_ir
        trainer.memory_rgb = memory_rgb

        del cluster_features_ir
        del cluster_features_rgb

        pseudo_labeled_dataset_ir = []

        ir_label = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):

            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
                ir_label.append(label.item())
        print('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))

        pseudo_labeled_dataset_rgb = []
        rgb_label = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                rgb_label.append(label.item())
        # pseudo_labeled_dataset_rgb ==>((fname, label.item(), cid))
        print('==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))

        ########################
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        height = args.height
        width = args.width

        train_transformer_rgb = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability=0.5)
        ])

        train_transformer_rgb1 = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability=0.5),
            ChannelExchange(gray=2)
        ])

        transform_thermal = T.Compose([
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((384, 128)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability=0.5),
            ChannelAdapGray(probability=0.5)])



        train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                              args.batch_size, args.workers, args.num_instances, iters,
                                              trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,
                                              train_transformer=transform_thermal)

        train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                                  args.batch_size // 2, args.workers, args.num_instances, iters,
                                                  trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,
                                                  train_transformer=train_transformer_rgb,
                                                  train_transformer1=train_transformer_rgb1)

        train_loader_ir.new_epoch()
        train_loader_rgb.new_epoch()

        trainer.train(epoch, train_loader_ir, train_loader_rgb, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader_ir))

        if epoch >= 0:
            ##############################
            args.test_batch = 64
            args.img_w = args.width
            args.img_h = args.height
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            transform_test = T.Compose([
                T.ToPILImage(),
                T.Resize((args.img_h, args.img_w)),
                T.ToTensor(),
                normalize,
            ])
            mode = 'all'
            data_path = data_dir
            query_img, query_label = process_test_regdb(data_path, trial=trial, modal='visible')
            # 函数的主要功能是处理 RegDB 数据集的测试数据，根据指定的模态（可见光或热红外）
            # 和试验编号，从对应的文本文件中读取测试数据的图像路径和标签信息，并将其返回。
            gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='thermal')

            gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
            nquery = len(query_label)
            ngall = len(gall_label)

            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)

            query_feat_fc = extract_query_feat(model, query_loader, nquery)
            # for trial in range(1):
            ngall = len(gall_label)
            gall_feat_fc = extract_gall_feat(model, gall_loader, ngall)
            # fc feature

            # distmat = calculate_similarity_matrix(query_feat_fc, gall_feat_fc)
            distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
            # 2060*2060
            cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)

            print('Test Trial: {}'.format(trial))
            print(
                'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

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
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


def main_worker_stage2(args, log_s1_name, log_s2_name):
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
    dataset_ir = get_data('regdb_ir', args.data_dir, trial=trial)
    dataset_rgb = get_data('regdb_rgb', args.data_dir, trial=trial)
    # Create model

    model = models.create(args.arch, img_size=(args.height, args.width), drop_path_rate=args.drop_path_rate0
                          , pretrained_path=args.pretrained_path, hw_ratio=args.hw_ratio, conv_stem=args.conv_stem,
                          cls_token_num=cls_token_num)
    model.cuda()
    model = nn.DataParallel(model)  # ,output_device=1)



    checkpoint = load_checkpoint(osp.join('./logs/' + log_s1_name + '/' + str(trial), 'model_best.pth.tar'))
    # checkpoint = load_checkpoint(osp.join('./logs/' + 'regdb_s2' + '/' + str(trial), 'model_best.pth.tar'))

    model.load_state_dict(checkpoint['state_dict'])


    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    # optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    optimizer = torch.optim.SGD(params, lr=args.lr1, momentum=0.9, weight_decay=args.weight_decay1)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)

    # Trainer
    trainer = ClusterContrastTrainer(model)

    for epoch in range(args.epochs):
        with torch.no_grad():
            if epoch == 0:
                # DBSCAN cluster
                ir_eps = 0.30
                print('IR Clustering criterion: eps: {:.3f}'.format(ir_eps))
                cluster_ir = DBSCAN(eps=ir_eps, min_samples=4, metric='precomputed', n_jobs=-1)
                rgb_eps = 0.30
                print('RGB Clustering criterion: eps: {:.3f}'.format(rgb_eps))
                cluster_rgb = DBSCAN(eps=rgb_eps, min_samples=4, metric='precomputed', n_jobs=-1)

            print('==> Create pseudo labels for unlabeled RGB data')

            cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width,
                                                 args.batch_size, args.workers,
                                                 testset=sorted(dataset_rgb.train))
            features_rgb,features_rgb_pa,ture_ids_rgb = extract_features(model, cluster_loader_rgb, print_freq=50, mode=1)

            # import csv
            # with open("features_rgb_first.csv", "w", newline="") as f:
            #     writer = csv.DictWriter(f, fieldnames=features_rgb.keys())
            #     writer.writeheader()
            #     writer.writerow(features_rgb)

            del cluster_loader_rgb,
            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)
            features_rgb_pa = torch.cat([features_rgb_pa[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)
            features_rgb_norm = F.normalize(features_rgb, dim=1)
            ture_ids_rgb = torch.cat([ture_ids_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)

            # np.save('features_rgb.npy', np.array(features_rgb))

            print('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width,
                                                args.batch_size, args.workers,
                                                testset=sorted(dataset_ir.train))
            features_ir, features_ir_pa, ture_ids_ir = extract_features(model, cluster_loader_ir, print_freq=50, mode=2)

            # import csv
            # with open("features_ir_first.csv", "w", newline="") as f:
            #     writer = csv.DictWriter(f, fieldnames=features_ir.keys())
            #     writer.writeheader()
            #     writer.writerow(features_ir)

            del cluster_loader_ir
            features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)
            features_ir_pa = torch.cat([features_ir_pa[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)
            ture_ids_ir = torch.cat([ture_ids_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)
            features_ir_norm = F.normalize(features_ir, dim=1)

            # np.save('features_ir.npy', np.array(features_ir))

            rerank_dist_ir = compute_jaccard_distance(features_ir_norm, k1=args.k1, k2=args.k2, search_option=3)
            pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
            rerank_dist_rgb = compute_jaccard_distance(features_rgb_norm, k1=args.k1, k2=args.k2, search_option=3)
            pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)
            del rerank_dist_rgb
            del rerank_dist_ir
            del features_ir_norm
            del features_rgb_norm

            pseudo_labels_rgb1 = torch.from_numpy(pseudo_labels_rgb)
            pseudo_labels_ir1 = torch.from_numpy(pseudo_labels_ir)
            print_intra_acc(ture_ids_rgb, ture_ids_ir, pseudo_labels_rgb1, pseudo_labels_ir1)



            num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)
            num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features, features_pa):
            centers = collections.defaultdict(list)
            centers_pa = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])
                centers_pa[labels[i]].append(features_pa[i])
                # centers_pa[labels[i]] = features_pa[i]

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

        # 处理-1标签归类
        cluster_features_ir, prototypes_ir = generate_cluster_features_corr(pseudo_labels_ir, features_ir)
        cluster_features_rgb, prototypes_rgb = generate_cluster_features_corr(pseudo_labels_rgb, features_rgb)
        print("Correcting label")
        pseudo_labels_ir_hat = correct_label(features_ir, pseudo_labels_ir, prototypes_ir)
        # pseudo_labels_ir_hat = pseudo_labels_ir
        pseudo_labels_rgb_hat = correct_label(features_rgb, pseudo_labels_rgb, prototypes_rgb)
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

        pseudo_labels_ir_update = update_label(features_ir, pseudo_labels_ir1, cluster_features_ir)
        pseudo_labels_rgb_update = update_label(features_rgb, pseudo_labels_rgb1, cluster_features_rgb)
        # del pseudo_labels_ir, pseudo_labels_rgb

        memory_ir = ClusterMemory(model.module.num_features*cls_token_num, num_cluster_ir, temp=args.temp,
                                  momentum=args.momentum, mode=args.memorybank, smooth=args.smooth,
                                  num_instances=args.num_instances).cuda()


        memory_rgb = ClusterMemory(model.module.num_features*cls_token_num, num_cluster_rgb, temp=args.temp,
                                   momentum=args.momentum, mode=args.memorybank, smooth=args.smooth,
                                   num_instances=args.num_instances).cuda()

        if args.memorybank == 'CM':
            memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
            memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()
        elif args.memorybank == 'CMhcl':
            memory_ir.features = F.normalize(cluster_features_ir.repeat(2, 1), dim=1).cuda()
            memory_rgb.features = F.normalize(cluster_features_rgb.repeat(2, 1), dim=1).cuda()
        trainer.memory_ir = memory_ir
        trainer.memory_rgb = memory_rgb



        pseudo_labeled_dataset_ir = []
        ir_label = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir_hat)):
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
                ir_label.append(label.item())
        print('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))

        pseudo_labeled_dataset_rgb = []
        rgb_label = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb_hat)):
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                rgb_label.append(label.item())

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
        result = ot.sinkhorn(a, b, M, reg=3, numItermax=5000, stopThr=1e-5)
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




        color_aug = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)  # T.
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        height = args.height
        width = args.width
        train_transformer_rgb = T.Compose([
            color_aug,
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability=0.5)
        ])

        train_transformer_rgb1 = T.Compose([
            color_aug,
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability=0.5),
            ChannelExchange(gray=2)
        ])

        transform_thermal = T.Compose([
            color_aug,
            T.Resize((height, width), interpolation=3),
            T.Pad(10),
            T.RandomCrop((384, 128)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalizer,
            ChannelRandomErasing(probability=0.5),
            ChannelAdapGray(probability=0.5)])


        train_loader_ir = get_train_loader_ir(args, dataset_ir, args.height, args.width,
                                              args.batch_size, args.workers, args.num_instances, iters,
                                              trainset=pseudo_labeled_dataset_ir, no_cam=args.no_cam,
                                              train_transformer=transform_thermal)

        train_loader_rgb = get_train_loader_color(args, dataset_rgb, args.height, args.width,
                                                  args.batch_size // 2, args.workers, args.num_instances, iters,
                                                  trainset=pseudo_labeled_dataset_rgb, no_cam=args.no_cam,
                                                  train_transformer=train_transformer_rgb,
                                                  train_transformer1=train_transformer_rgb1)

        train_loader_ir.new_epoch()
        train_loader_rgb.new_epoch()


        trainer.train(epoch, train_loader_ir, train_loader_rgb, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader_ir), i2r=i2r, r2i=r2i,i2r_p=i2r_p, r2i_p=r2i_p)

        if epoch >= 0:
            ##############################
            args.test_batch = 64
            args.img_w = args.width
            args.img_h = args.height
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            transform_test = T.Compose([
                T.ToPILImage(),
                T.Resize((args.img_h, args.img_w)),
                T.ToTensor(),
                normalize,
            ])

            data_path = data_dir
            query_img, query_label = process_test_regdb(data_path, trial=trial, modal='visible')
            gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='thermal')

            gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
            nquery = len(query_label)
            ngall = len(gall_label)
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
            query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
            query_feat_fc = extract_query_feat(model, query_loader, nquery)
            # for trial in range(1):
            ngall = len(gall_label)
            gall_feat_fc = extract_gall_feat(model, gall_loader, ngall)
            # fc feature

            distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))

            cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)

            print('Test Trial: {}'.format(trial))
            print(
                'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

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
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='rgedb_rgb',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-j', '--workers', type=int, default=0)
    parser.add_argument('--height', type=int, default=384, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=8,
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
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.8,
                        help="update momentum for the hybrid memory")
    parser.add_argument('-mb', '--memorybank', type=str, default='CMhcl',
                        choices=['CM', 'CMhcl'])
    parser.add_argument('--smooth', type=float, default=0, help="label smoothing")

    # optimizer
    parser.add_argument('--lr0', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--lr1', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data/RegDB/'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--no-cam', action="store_true")

    # vit

    parser.add_argument('--weight-decay0', type=float, default=5e-4)
    parser.add_argument('--weight-decay1', type=float, default=5e-4)


    parser.add_argument('--drop-path-rate0', type=float, default=0.3)
    parser.add_argument('--drop-path-rate1', type=float, default=0.3)
    parser.add_argument('--hw-ratio', type=int, default=2)
    parser.add_argument('--self-norm', action="store_true")
    parser.add_argument('--conv-stem', action="store_true")

    parser.add_argument('--cls-token-num', type=int, default=6)
    parser.add_argument('-pp', '--pretrained-path', type=str, default='vit_base_ics_cfs_lup.pth')
    main()
