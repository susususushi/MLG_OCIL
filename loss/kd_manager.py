# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
import random
from copy import deepcopy


def get_old_sample(mem_x, mem_y, old_cls_label):
    old_idx = torch.where(mem_y < old_cls_label)
    x = mem_x[old_idx].clone()
    return x


class kd_manager:
    def __init__(self, params):
        self.params = params
        self.T = params.T
        self.teacher_model = None
        self.kd_type = params.kd_type
        self.ckd = Common_KD(params)
        self.lamda = params.kd_lamda
        self.dist = DIST(params)
        self.fkd = Cosine_Feature()
        if params.old_cls and (not params.fix_order):
            raise NotImplementedError('old only cannot be used when fixed order is False')

    def update_teacher(self, model):
        self.teacher_model = deepcopy(model).cuda()
        self.teacher_model.eval()
        # Freeze all parameters from the model, including the heads
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def get_loss(self, x, y, task_id, model):
        if self.teacher_model is None: return 0.0

        loss = 0.0
        if self.kd_type == 'fkd' and self.params.old_cls:
            old_cls_label = self.params.num_classes // self.params.num_tasks * task_id
            x = get_old_sample(x, y, old_cls_label)
            if x.size(0) == 0: return 0.0

        if self.kd_type == 'fkd':
            with torch.no_grad():
                old_feat = self.teacher_model.features(x)
            new_feat = model.features(x)

            if self.params.bf:
                bf_old_feat = self.teacher_model.BatchFormer(old_feat.unsqueeze(1)).squeeze(1)
                bf_new_feat = model.BatchFormer(new_feat.unsqueeze(1)).squeeze(1)
                loss = loss + self.fkd(bf_old_feat, bf_new_feat)

            loss = loss + self.fkd(old_feat, new_feat)

            if self.params.old_cls:
                old_pro = self.teacher_model.linear.weight[:old_cls_label]
                new_pro = model.linear.weight[:old_cls_label]
            else:
                old_pro = self.teacher_model.linear.weight
                new_pro = model.linear.weight

            loss = loss + self.fkd(old_pro, new_pro)
            return loss * self.lamda

        logits = model(x)
        with torch.no_grad():
            old_logits = self.teacher_model(x)

        if self.params.old_cls:
            # count old cls id since classes' order is fixed, i.e. [0,1],[2,3], ...
            old_cls_label = self.params.num_classes // self.params.num_tasks * task_id
            logits = logits[:, :old_cls_label]
            old_logits = old_logits[:, :old_cls_label]

        if self.kd_type == 'ckd':
            loss = loss + self.ckd(logits, old_logits)
        elif self.kd_type == 'dist':
            loss = loss + self.dist(logits, old_logits)
        else:
            raise NotImplementedError(
                'undefined kd_type {}'.format(self.kd_type))

        # multiply temperature
        if self.kd_type != 'dist':
            loss = loss * (self.T ** 2)
        return loss * self.lamda


class Cosine_Feature(nn.Module):
    def __init__(self):
        super(Cosine_Feature, self).__init__()

    def forward(self, old_feat, new_feat):
        old_feat = nn.functional.normalize(old_feat, 2, 1)
        new_feat = nn.functional.normalize(new_feat, 2, 1)

        y = torch.ones(new_feat.shape[0]).cuda()
        criterion = nn.CosineEmbeddingLoss(reduction='mean')
        return criterion(old_feat, new_feat, y)


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DIST(nn.Module):
    # def __init__(self, beta=1.0, gamma=1.0):
    def __init__(self, params):
        super(DIST, self).__init__()
        # self.beta = beta
        # self.gamma = gamma

    def forward(self, z_s, z_t):
        y_s = z_s.softmax(dim=1)
        y_t = z_t.softmax(dim=1)
        inter_loss = inter_class_relation(y_s, y_t)
        intra_loss = intra_class_relation(y_s, y_t)
        # kd_loss = self.beta * inter_loss + self.gamma * intra_loss
        # return inter_loss
        kd_loss = inter_loss + intra_loss
        return kd_loss


class Common_KD(nn.Module):
    def __init__(self, params):
        super(Common_KD, self).__init__()
        self.T = params.T

    def forward(self, logits, old_logits):
        log_scores_norm = F.log_softmax(logits / self.T, dim=1)
        targets_norm = F.softmax(old_logits / self.T, dim=1)
        # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
        kd_loss = (-1.0 * targets_norm * log_scores_norm).sum(dim=1).mean()
        return kd_loss
