# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import random
import numpy as np
import argparse
import torch.nn.functional as F

input_size_match = {
    'cifar10': (3, 32, 32),
    'cifar100': (3, 32, 32),
    'mini_imagenet': (3, 84, 84)
}
feature_size_match = {
    'cifar100': 160,
    'cifar10': 160,
    'mini_imagenet': 640,
}

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
        # self.feat_size = 3
        #这里索引就是对应类别了,方便forward
        self.mean_feat = torch.zeros((params.num_classes, self.feat_size)).float().cuda()
        self.used = torch.zeros((params.num_classes,)).long().cuda()
        self.all_feat_label = None
        self.all_feat = None

    def update(self, prototype, class_label):
        #here copy from reservoir random, prototypes there have been normalized 
        self.mean_feat[class_label] = prototype # transform it in order
        self.used.fill_(0)
        self.used[class_label] = 1
        self.all_feat_label = torch.where(self.used > 0)[0]
        self.all_feat = self.mean_feat[self.all_feat_label]
        self.all_feat = nn.functional.normalize(self.all_feat, p=2, dim=1)

    def forward(self, batch_x, batch_y, model):#20231020
        old_cls_idx = torch.where(self.used[batch_y] > 0)[0]
        if old_cls_idx.size(0) == 0:
            return 0.0
        
        old_cls_sample = batch_x[old_cls_idx]
        old_cls_feat = model.features(old_cls_sample)
        # old_cls_feat = batch_x[old_cls_idx]
        old_cls_feat = nn.functional.normalize(old_cls_feat, p=2, dim=1)

        label = batch_y[old_cls_idx]

        #one hot label
        target = (label.unsqueeze(1) == self.all_feat_label).bool()
        # score shape = (old_cls_feat.size(0), self.all_feat.size(0))
        score = torch.div(
            torch.matmul(old_cls_feat, self.all_feat.T), self.supcon_temperature
        )
        #calculate the probability
        score = torch.exp(score)
        prob = torch.div(score[target], score.sum(1))
        #compute final cross entropy loss
        loss = -torch.log(prob).mean()
        return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online Continual Learning PyTorch")
    ########################General#########################
    parser.add_argument('--seed', dest='seed', default=0, type=int,
                        help='Random seed, if seed < 0, it will not be set')
    parser.add_argument('--data', dest='data', default="cifar10", type=str,
                        choices=['cifar10', 'cifar100', 'mini_imagenet'],
                        help='Path to the dataset. (default: %(default)s)')
    
    parser.add_argument('--run_time', dest='run_time', default=10, type=int,
                        help='the time of running')

    parser.add_argument('--review_trick', dest='review_trick', default=False, type=boolean_string,
                        help='whethre use review trick')
    
    parser.add_argument('--agent', dest='agent', default="er", type=str,
                        choices = ["er", "scr"],
                        help='agent')
    
    parser.add_argument('--retrieve', dest='retrieve', default="random", type=str,
                        choices = ["random", "mfd", 'pearson', 'mir'],
                        help='retrieve method')
    
    parser.add_argument('--data_aug', dest='data_aug', default=False, type=boolean_string,
                        help='whethre use data augmentation')
    
    parser.add_argument('--bf', dest='bf', default=False, type=boolean_string,
                        help='whethre use BatchFormer')
    
    parser.add_argument('--dist', dest='dist', default=False, type=boolean_string,
                        help='whether use distlinear classifier')
    
    parser.add_argument('--drop_rate', dest='drop_rate', default=0.25, type=float,
                        help='the rate of drop out in BatchFormer')
    
    ########################temperature for scr and normalized classifier#########################
    parser.add_argument('--temperature', dest='temperature', default=0.1, type=float,
                        help='the temperature of supervised contrastive learning')

    parser.add_argument('--num_classes', dest='num_classes', default=10,
                        type=int,
                        help='Number of classes in total')
    
    parser.add_argument('--supcon_temperature', dest='supcon_temperature', default=0.1,
                        type=float,
                        help='supcon_temperature of feature distillation')
    
    args = parser.parse_args()

    initial(args)
    targets = torch.tensor([0,3,2]).long().cuda()
    feature = torch.tensor([[0,1,2],
                      [1,1,1],
                      [2,4,8]]).float().cuda()

    fkd = Feature_KD(args)
    fkd.update(feature, targets)

    targets = torch.tensor([0,5,2]).long().cuda()
    features = torch.tensor([[2,3,4],
                      [7,1,5],
                      [16,32,64]]).float().cuda()
    print(fkd(features, targets))
    # print("np log : ", np.log(0.5))
    # print("torch log : ", torch.log(torch.tensor([0.5])))
    # fkd = Feature_KD(args)
    # pro = torch.randn((5,feature_size_match[args.data])).float().cuda()
    # lb = torch.randint(0, args.num_classes, size=(5,)).long().cuda()
    # print(lb)
    # lb = torch.tensor([0,1,1,5,6,0,8,8,9]).long().cuda()
    # a = torch.tensor([  [1,2,3,4,5],
    #                     [5,4,1,2,3],
    #                     ])
    # b = torch.tensor([5,4,1,2,3])
    # c = torch.tensor([0,0,0,0,0])
    # c[b-1] = a
    # print(c)

    # fkd.update(pro, lb)
    # print(fkd.used)
    # pro = torch.randn((5,feature_size_match[args.data])).float().cuda()
    # lb = torch.randint(0, args.num_classes, size=(5,)).long().cuda()
    # lb = torch.Tensor([1,1,1,1,1]).long().cuda()
    # print(lb)
    # print(fkd(pro,lb))



