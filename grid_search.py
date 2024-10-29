# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:33:37 2022

@author: czjghost
"""


import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import sys
from ER import experience_replay
from SCR import supervised_contrastive_replay
from ER_BatchFormer import experience_replay_BatchFormer
from logger import Logger, build_holder
import matplotlib
from itertools import product
import os
matplotlib.use('Agg')

n_classes = {
    'cifar100': 100,
    'cifar10': 10,
    'mini_imagenet': 100,
}

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def parameter_setting():
    #mainly reference from https://github.com/RaptorMai/online-continual-learning
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Online Continual Learning PyTorch")
    ########################General#########################
    parser.add_argument('--seed', dest='seed', default=0, type=int,
                        help='Random seed, if seed < 0, it will not be set')
    
    #选择的数据集，输入对应的路径, 论文中常用前三种
    parser.add_argument('--data', dest='data', default="cifar10", type=str,
                        choices=['cifar10', 'cifar100', 'mini_imagenet'],
                        help='Path to the dataset. (default: %(default)s)')
    
    parser.add_argument('--run_time', dest='run_time', default=10, type=int,
                        help='the time of running')

    parser.add_argument('--review_trick', dest='review_trick', default=False, type=boolean_string,
                        help='whethre use review trick')
    
    parser.add_argument('--agent', dest='agent', default="er_bf", type=str,
                        choices = ["er", "er_bf", "scr"],
                        help='agent')
    
    parser.add_argument('--data_aug', dest='data_aug', default=False, type=boolean_string,
                        help='whethre use data augmentation')
    
    parser.add_argument('--bf', dest='bf', default=False, type=boolean_string,
                        help='whethre use BatchFormer')
    
    parser.add_argument('--feat_lam', dest='feat_lam', default=0.25, type=float,
                        help='the lambda of feature knowledge distillation')
    
    parser.add_argument('--drop_rate', dest='drop_rate', default=0.5, type=float,
                        help='the rate of drop out in BatchFormer')
    
    parser.add_argument('--crop_rate', dest='crop_rate', default=0.85, type=float,
                        help='the rate of RandomCrop')
    
    ########################temperature for scr and normalized classifier#########################
    parser.add_argument('--temperature', dest='temperature', default=0.1, type=float,
                        help='the temperature of supervised contrastive learning')
    
    ########################Revised Focal Loss#########################
    #default: rfl_alpha=0.25,rfl_sigma=0.5,rfl_miu=0.3
    parser.add_argument('--rfl_alpha', dest='rfl_alpha', default=0.25, type=float,
                        help='Revised Focal loss alpha')
    
    parser.add_argument('--rfl_sigma', dest='rfl_sigma', default=0.5, type=float,
                        help='Revised Focal loss sigma')
    
    parser.add_argument('--rfl_miu', dest='rfl_miu', default=0.3, type=float,
                        help='Revised Focal loss miu')
    
    ########################Revised Focal Loss#########################
    parser.add_argument('--kal_alpha', dest='kal_alpha', default=0.25, type=float,
                        help='KeepAwayLoss alpha')
    
    ########################virtual knowledge distillation#########################
    parser.add_argument('--kd_trick', dest='kd_trick', default=False, type=boolean_string,
                        help='whethre use knowledge distillation loss')
    #vkd 
    parser.add_argument('--cor_prob', dest='cor_prob', default=0.99, type=float,
                        help='the correct probability of target class in virtual kd')
    #kd_lamda * kd_loss
    parser.add_argument('--kd_lamda', dest='kd_lamda', default=0.1, type=float,
                        help='lamda used in knowledge distillation loss')
    
    parser.add_argument('--T', dest='T', default=2.0, type=float,
                        help='temperature for Distillation loss function, orginal paper set to 2, vkd is set to 20')
    #comparison between vkd and ckd, ckd is common knowledge distillation which is used in most CL scenario
    parser.add_argument('--kd_type', dest='kd_type', default="vkd", type=str,
                        choices=["ckd","vkd"],
                        help='which type of kd will be used')    
    
    ########################Experience Replay#########################
    parser.add_argument('--loss', dest='loss', default="ce", type=str,
                        choices = ["ce","focal","rfocal","scl"],
                        help='selected loss')
    
    #the strategy of ncm is same as SCR
    parser.add_argument('--classify', dest='classify', default="max", type=str,
                        choices = ["ncm", "max"],
                        help='selected classification')
    
    ########################Focal Loss#########################
    #default : alpha=0.25, gamma=2.0
    parser.add_argument('--focal_alpha', dest='focal_alpha', default=0.25, type=float,
                        help='Focal loss alpha')
    
    parser.add_argument('--focal_gamma', dest='focal_gamma', default=2.0, type=float,
                        help='Focal loss gamma')
    
    ########################Optimizer#########################
    parser.add_argument('--optimizer', dest='optimizer', default='SGD', type=str,
                        choices=['SGD', 'Adam'],
                        help='Optimizer (default: %(default)s)')

    parser.add_argument('--learning_rate', dest='learning_rate', default=0.1,
                        type=float,
                        help='Learning_rate of models (default: %(default)s)')

    parser.add_argument('--batch', dest='batch', default=10,
                        type=int,
                        help='Batch size (default: %(default)s)')

    parser.add_argument('--eps_mem_batch', dest='eps_mem_batch', default=10,
                        type=int,
                        help='the number of sample selected per batch from memory (default: %(default)s)')

    parser.add_argument('--mem_size', dest='mem_size', default=1000,
                        type=int,
                        help='Memory buffer size (default: %(default)s)')    
    #note that if last batch <= test_batch, then it will be the size of last batch
    parser.add_argument('--test_batch', dest='test_batch', default=128,
                        type=int,
                        help='Test batch size (default: %(default)s)')

    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0,
                        help='weight_decay')

    ########################Data#########################
    parser.add_argument('--num_tasks', dest='num_tasks', default=5,
                        type=int,
                        help='Number of tasks (default: %(default)s), OpenLORIS num_tasks is predefined')
    
    parser.add_argument('--num_classes', dest='num_classes', default=10,
                        type=int,
                        help='Number of classes in total')

    parser.add_argument('--fix_order', dest='fix_order', default=True,
                        type=bool,
                        help='In NC scenario, should the class order be fixed (default: %(default)s)')
    
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    
    return args

def initial(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
def multiple_run(args, holder, log):
    #check whether let seed go
    if args.seed >= 0:
        initial(args)
    
    #set initial total tasks
    if args.data == 'cifar10':
        args.num_tasks = 5
    elif args.data == 'mini_imagenet':
        args.num_tasks = 10 
    elif args.data == 'cifar100':
        args.num_tasks = 10 
    
    args.num_classes = n_classes[args.data]
    
    if args.agent == "er":
        return experience_replay(args, holder, log)
    elif args.agent == "scr":
        return supervised_contrastive_replay(args, holder, log)
    elif args.agent == "er_bf":
        return experience_replay_BatchFormer(args, holder, log)
    else:
        raise NotImplementedError(
                    'agent not supported: {}'.format(args.agent))

def tune_holder(args):
    holder = ""
    if args.seed >= 0:
        holder = "result/" + "seed=" + str(args.seed) + "/" 
        
    holder = holder + "BF" + "_" + args.data + "_" + args.classify
    holder = holder + "_" + args.loss

    holder = holder + "_drop=" + str(args.drop_rate)
            
    holder = holder + "_τ=" + str(args.temperature)    
    holder = holder + "_feat_λ=" + str(args.feat_lam)
    
    holder = holder + "_crop=" + str(args.crop_rate)
    
    if args.data_aug:
        holder = holder + "_aug"
        
    holder = holder + "_eps=" + str(args.eps_mem_batch)
    holder = holder + "_mem=" + str(args.mem_size)
    holder = holder + "_lr=" + str(args.learning_rate) 

    if args.fix_order:
        holder = holder + "_fix"
    
    if not os.path.exists(holder):
        os.makedirs(holder)
        
    return holder

if __name__ == "__main__":
    #all config

    args = parameter_setting()
#    drop_rate 和 feat_lam的  0.1 留到最后再跑
#    datasets = ['cifar10', 'cifar100', 'mini_imagenet']
    data = args.data
    
    batch_size = [16, 32, 64]
    drop_rate = [0.25, 0.5]
    crop_rate = [0.75, 0.85]
    feat_lam = [0.0, 0.25, 0.5, 1.0]
    tt = [0.1, 0.15]
    
    best_acc = {'cifar10': 0.0, 'cifar100': 0.0, 'mini_imagenet': 0.0}
    best_params = {'cifar10': None, 'cifar100': None, 'mini_imagenet': None}
    
    
    tune_tuple = product(batch_size, drop_rate, crop_rate, feat_lam, tt)
    for datas in tune_tuple:
        bs, dr, cr, fl, tem = datas
        print("dataset : ", data, ", cur params: ", datas)
        print('\n')
        args.eps_mem_batch = bs
        args.drop_rate = dr
        args.crop_rate = cr
        args.feat_lam = fl
        args.temperature = tem
        
        holder = tune_holder(args)
        
        log = Logger(holder + "/running_result.log", stream=sys.stdout)
        sys.stdout = log
        sys.stderr = log
        avr_end_acc = multiple_run(args, holder, log)
        
        if best_acc[data] < avr_end_acc[0]:
            best_acc[data] = avr_end_acc[0]
            best_params[data] = datas
        break
    
    best_path = "best_result/"
    if not os.path.exists(best_path):
        os.makedirs(best_path)
    print("\n\n####################Final Best Parameters####################")
    print("(batch_size drop_rate crop_rate feat_lam tt)")
    if best_params['cifar10'] is not None:
        print(best_params['cifar10'])
        print('best_acc = ', best_acc['cifar10'])
        print('\n')
        with open(best_path + "/cifar10.txt", "w") as f:
            f.write("(batch_size drop_rate crop_rate feat_lam tt)\n")
            f.write(str(best_params['cifar10']))
            f.write('\n')
            f.write('best_acc = ' + str(best_acc['cifar100']))
            f.write('\n')
        
    
    if best_params['cifar100'] is not None:
        print(best_params['cifar100'])
        print('best_acc = ', best_acc['cifar100'])
        print('\n')
        with open(best_path + "/cifar100.txt", "w") as f:
            f.write("(batch_size drop_rate crop_rate feat_lam tt)\n")
            f.write(str(best_params['cifar100']))
            f.write('\n')
            f.write('best_acc = ' +  str(best_acc['cifar100']))
            f.write('\n')
        
    if best_params['mini_imagenet'] is not None: 
        print(best_params['mini_imagenet'])
        print('best_acc = ', best_acc['mini_imagenet'])
        print('\n')
        with open(best_path + "/mini_imagenet.txt", "w") as f:
            f.write("(batch_size drop_rate crop_rate feat_lam tt)\n")
            f.write(str(best_params['mini_imagenet']))
            f.write('\n')
            f.write('best_acc = ' + str(best_acc['mini_imagenet']))
            f.write('\n')
    
    
