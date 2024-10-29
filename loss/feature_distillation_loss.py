# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import random
import numpy as np
import argparse
import torch.nn.functional as F
from setup_elements import input_size_match, feature_size_match


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def initial(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Feature_KD(nn.Module):
    def __init__(self, params):
        super(Feature_KD, self).__init__()
        self.params = params
        self.supcon_temperature = params.supcon_temperature
        self.feat_size = feature_size_match[params.data]
        # The index is the corresponding class, convenient forward
        self.mean_feat = torch.zeros((params.num_classes, self.feat_size)).float().cuda()
        self.used = torch.zeros((params.num_classes,)).long().cuda()
        self.all_feat_label = None
        self.all_feat = None

    def update(self, prototype, class_label):
        # here copy from reservoir random, prototypes there have been normalized
        self.mean_feat[class_label] = prototype  # transform it in order
        self.used.fill_(0)
        self.used[class_label] = 1
        self.all_feat_label = torch.where(self.used > 0)[0]
        self.all_feat = self.mean_feat[self.all_feat_label]

    def forward(self, batch_x, batch_y, model):  # 20230618
        old_cls_idx = torch.where(self.used[batch_y] > 0)[0]
        if old_cls_idx.size(0) == 0:
            return 0.0

        old_cls_sample = batch_x[old_cls_idx]
        old_cls_feat = model.features(old_cls_sample)
        old_cls_feat = nn.functional.normalize(old_cls_feat, p=2, dim=1)

        label = batch_y[old_cls_idx]

        y = torch.ones(old_cls_feat.shape[0]).cuda()  # 1 - cos(a,b)
        criterion = nn.CosineEmbeddingLoss(reduction='mean')

        return criterion(self.mean_feat[label], old_cls_feat, y)
