# -*- coding: utf-8 -*-

import torch
import os
import numpy as np
import torch.nn as nn
from ipdb import set_trace
from torchvision import transforms
from copy import deepcopy
import copy
import random
import time
import torch.nn.functional as F
from setup_elements import input_size_match, feature_size_match


def initial(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def get_grad_vector(pp, grad_dims):
    """
        gather the gradients in one vector
    """
    grads = torch.Tensor(sum(grad_dims)).cuda()
    grads.fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1
    return grads


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * (x @ y.t())

    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def get_grad_vector(pp, grad_dims):
    """
        gather the gradients in one vector
    """
    grads = torch.Tensor(sum(grad_dims)).cuda()
    grads.fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1
    return grads


class Reservoir_Random(object):

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.cur_idx = 0
        self.n_class_seen = 0
        self.cls_set = {}
        self.all_classes = params.num_classes
        self.accumulate_cnt = torch.zeros((params.num_classes,)).long().cuda()

        # for pearson retrieve
        self.subsample = params.subsample

        self.buffer_size = params.mem_size
        self.input_size = input_size_match[params.data]
        self.feat_size = feature_size_match[params.data]

        self.buffer_img = torch.zeros((self.buffer_size,) + self.input_size).float().cuda()
        self.buffer_label = torch.zeros(self.buffer_size).long().cuda()

        # used for Reservoir Sampling counts
        self.n_sample_seen_so_far = 0

        # accumulated feature
        self.acc_mean_feat = torch.zeros((0, self.feat_size)).float().cuda()
        # mean feature and after normalization
        self.mean_feat = torch.zeros((0, self.feat_size)).float().cuda()
        self.mean_feat_label = torch.zeros(0).long().cuda()

        # self.all_feats_dict = {}  # save all_feat for each class
        # self.mean_feats_dict = {}  # save mean_feat for each class

    def get_future_step_parameters(self, model, grad_vector, grad_dims):
        """
        computes \theta-\delta\theta
        :param this_net:
        :param grad_vector:
        :return:
        """
        new_model = copy.deepcopy(model)
        self.overwrite_grad(new_model.parameters, grad_vector, grad_dims)
        with torch.no_grad():
            for param in new_model.parameters():
                if param.grad is not None:
                    param.data = param.data - self.params.learning_rate * param.grad.data
        return new_model

    def overwrite_grad(self, pp, new_grad, grad_dims):
        """
            This is used to overwrite the gradients with a new gradient
            vector, whenever violations occur.
            pp: parameters
            newgrad: corrected gradient
            grad_dims: list storing number of parameters at each layer
        """
        cnt = 0
        for param in pp():
            param.grad = torch.zeros_like(param.data)
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = new_grad[beg: en].contiguous().view(
                param.data.size())
            param.grad.data.copy_(this_grad)
            cnt += 1

    def certain_filter(self, model, batch_x, batch_y):
        logits_pre = model(batch_x)
        loss = F.cross_entropy(logits_pre, batch_y)
        loss.backward()

        grad_dims = []
        for param in model.parameters():
            grad_dims.append(param.data.numel())
        grad_vector = get_grad_vector(model.parameters, grad_dims)
        model_temp = self.get_future_step_parameters(model, grad_vector, grad_dims)
        if batch_x.size(0) > self.params.filter_keep:
            with torch.no_grad():
                # logits_pre = model(batch_x)
                logits_post = model_temp(batch_x)
                # softmax operation
                logits_pre = F.softmax(logits_pre, 1)
                logits_post = F.softmax(logits_post, 1)

                ohe_label = F.one_hot(batch_y, num_classes=logits_pre.size(1)).bool().cuda()

                tar_logit_pre = (logits_pre[ohe_label].view(-1, 1)) * (torch.ones_like(logits_pre).cuda())
                tar_logit_post = (logits_post[ohe_label].view(-1, 1)) * (torch.ones_like(logits_post).cuda())

                var_pre = ((tar_logit_pre - logits_pre) ** 2).mean(1)
                var_post = ((tar_logit_post - logits_post) ** 2).mean(1)

                # print("present = ", var_pre)
                # print("future = ", var_post)

                scores = var_post - var_pre  # the target logit is more higher, the var is more big
                big_ind = scores.sort(descending=True)[1][:self.params.filter_keep]

            return batch_x[big_ind], batch_y[big_ind]
        else:
            return batch_x, batch_y

    def is_empty(self):
        if self.cur_idx > 0:
            return False
        return True

    def update(self, x_train, y_train):

        n = x_train.size(0)
        for i in range(n):
            if self.cur_idx < self.buffer_size:
                self.buffer_img[self.cur_idx] = x_train[i]

                self.buffer_label[self.cur_idx] = y_train[i]
                self.cur_idx += 1

                if int(y_train[i]) in self.cls_set:
                    self.cls_set[int(y_train[i])] += 1
                else:
                    self.cls_set[int(y_train[i])] = 1
                    self.n_class_seen += 1

            else:
                r_idx = np.random.randint(0, self.n_sample_seen_so_far + i)

                if r_idx < self.buffer_size:

                    self.cls_set[int(self.buffer_label[r_idx])] -= 1

                    if self.cls_set[int(self.buffer_label[r_idx])] == 0:
                        self.cls_set.pop(int(self.buffer_label[r_idx]))
                        self.n_class_seen -= 1

                    self.buffer_img[r_idx] = x_train[i]
                    self.buffer_label[r_idx] = y_train[i]

                    if int(y_train[i]) in self.cls_set:
                        self.cls_set[int(y_train[i])] += 1
                    else:
                        self.cls_set[int(y_train[i])] = 1
                        self.n_class_seen += 1

            # update total number of sample have seen so far
            self.n_sample_seen_so_far += 1

    def get_future_step_parameters(self, model, grad_vector, grad_dims):
        """
        computes \theta-\delta\theta
        :param this_net:
        :param grad_vector:
        :return:
        """
        new_model = deepcopy(model)
        self.overwrite_grad(new_model.parameters, grad_vector, grad_dims)
        with torch.no_grad():
            for param in new_model.parameters():
                if param.grad is not None:
                    param.data = param.data - self.params.learning_rate * param.grad.data
        return new_model

    def overwrite_grad(self, pp, new_grad, grad_dims):
        """
            This is used to overwrite the gradients with a new gradient
            vector, whenever violations occur.
            pp: parameters
            newgrad: corrected gradient
            grad_dims: list storing number of parameters at each layer
        """
        cnt = 0
        for param in pp():
            param.grad = torch.zeros_like(param.data)
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = new_grad[beg: en].contiguous().view(param.data.size())
            param.grad.data.copy_(this_grad)
            cnt += 1

    # The incoming batch has been backward when retrieved
    def MIR_Retrieve(self, batch_size, model):
        sub_x, sub_y = self.Random_Retrieve(self.subsample)
        grad_dims = []
        for param in model.parameters():
            grad_dims.append(param.data.numel())
        grad_vector = get_grad_vector(model.parameters, grad_dims)
        model_temp = self.get_future_step_parameters(model, grad_vector, grad_dims)
        if sub_x.size(0) > 0:
            with torch.no_grad():
                if self.params.retrieve == 'pearson':
                    logits_pre = model.forward(sub_x)
                    logits_post = model_temp.forward(sub_x)
                    scores = pearson_correlation(logits_post, logits_pre)

                elif self.params.retrieve == 'mfd':
                    crit = nn.MSELoss(reduction='none')
                    feature_pre = model.features(sub_x)
                    feature_post = model_temp.features(sub_x)
                    scores = crit(feature_post, feature_pre).mean(1)

                else:
                    logits_pre = model.forward(sub_x)
                    logits_post = model_temp.forward(sub_x)
                    pre_loss = F.cross_entropy(logits_pre, sub_y, reduction='none')
                    post_loss = F.cross_entropy(logits_post, sub_y, reduction='none')
                    scores = post_loss - pre_loss

                big_ind = scores.sort(descending=True)[1][:min(batch_size, sub_y.size(0))]
            return sub_x[big_ind], sub_y[big_ind]
        else:
            return sub_x, sub_y

    def Random_Retrieve(self, batch_size):

        all_index = np.arange(self.cur_idx)

        select_batch_size = min(batch_size, self.cur_idx)
        select_index = torch.from_numpy(np.random.choice(all_index, select_batch_size, replace=False)).long().cuda()

        x = self.buffer_img[select_index]
        y = self.buffer_label[select_index]
        return x, y

    def retrieve(self, batch_size, model=None):
        if self.cur_idx <= batch_size:
            all_index = torch.arange(self.cur_idx).cuda()
            x, y = self.buffer_img[all_index], self.buffer_label[all_index]
            return x, y

        x, y = None, None
        if self.params.retrieve == 'random':
            x, y = self.Random_Retrieve(batch_size)
        elif self.params.retrieve == 'mfd' or self.params.retrieve == 'pearson' or self.params.retrieve == 'mir':
            x, y = self.MIR_Retrieve(batch_size, model)
        else:
            raise NotImplementedError(
                'retrieve method not supported: {}'.format(self.params.retrieve))
        return x, y

    def accumulate_update_prototype(self, model, task_id):
        # model.eval()
        classes = torch.tensor(list(self.cls_set.keys()))
        n = classes.size(0)
        for i in range(n):
            idx = (self.buffer_label == classes[i]).nonzero(as_tuple=False).flatten()
            all_img = self.buffer_img[idx]

            with torch.no_grad():
                if torch.cuda.is_available():
                    all_img = all_img.cuda()

                all_feat = model.features(all_img)
                all_feat = nn.functional.normalize(all_feat, p=2, dim=1)
                tot = len(idx)

                # save all_feat
                # self.all_feats_dict[classes[i].item()] = all_feat.cpu()

                # calculating mean feature
                current_feat = all_feat.sum(0) / tot
                if self.accumulate_cnt[classes[i]].item() == 0:
                    # not appear
                    self.acc_mean_feat = torch.cat([self.acc_mean_feat, current_feat.unsqueeze(0)], dim=0)  # accumulate mean feature
                    self.mean_feat_label = torch.cat([self.mean_feat_label, classes[i:i + 1].cuda()], dim=0)
                    self.accumulate_cnt[classes[i]] = tot
                    # calculating mean feature after normalization
                    norm_feat = nn.functional.normalize(current_feat, 2, 0)
                    self.mean_feat = torch.cat([self.mean_feat, norm_feat.unsqueeze(0)], dim=0)  # normalized accumulate mean feature
                    # self.mean_feats_dict[classes[i].item()] = current_feat.cpu() # save mean_feat
                else:
                    # find feature indice where class[i] belongs to
                    indice = torch.where(self.mean_feat_label == classes[i])[0]
                    assert self.mean_feat_label[indice] == classes[i]
                    now_tot = self.accumulate_cnt[classes[i]] + tot
                    self.acc_mean_feat[indice] = current_feat * (tot / now_tot) + self.acc_mean_feat[indice] * (self.accumulate_cnt[classes[i]] / now_tot)
                    self.accumulate_cnt[classes[i]] = now_tot
                    # calculating mean feature after normalization
                    self.mean_feat[indice] = nn.functional.normalize(self.acc_mean_feat[indice], 2, 0)
                    # self.mean_feats_dict[classes[i].item()] = self.acc_mean_feat[indice].cpu()

        # torch.save(self.all_feats_dict, '{}_all_feats.pt'.format(task_id))
        # torch.save(self.mean_feats_dict, '{}_mean_feats.pt'.format(task_id))
        # self.all_feats_dict.clear()
        # self.mean_feats_dict.clear()
        # model.train()

    def update_prototype(self, model):
        model.eval()
        classes = torch.tensor(list(self.cls_set.keys()))

        n = classes.size(0)

        self.mean_feat = torch.zeros((n, self.feat_size)).float().cuda()
        self.mean_feat_label = torch.zeros(n).long().cuda()

        for i in range(n):
            idx = (self.buffer_label == classes[i]).nonzero(as_tuple=False).flatten()

            self.mean_feat_label[i] = classes[i]
            all_img = self.buffer_img[idx]

            with torch.no_grad():
                if torch.cuda.is_available():
                    all_img = all_img.cuda()

                all_feat = model.features(all_img).data
                all_feat = nn.functional.normalize(all_feat, p=2, dim=1)
                self.mean_feat[i] = all_feat.sum(0) / len(idx)
                # add additional normalize after calculating mean feature,20230616
                self.mean_feat[i] = nn.functional.normalize(self.mean_feat[i], p=2, dim=0)
        model.train()
        print("update total of {} classes have update their NCM".format(n))

    def classify_mat(self, x_test, y_test=None):

        dist = euclidean_dist(self.mean_feat, x_test)
        m_idx = torch.argmin(dist, 0)
        pre = self.mean_feat_label[m_idx]
        return pre
