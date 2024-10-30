import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import sys
from SCR import supervised_contrastive_replay
from ER import experience_replay
from logger import Logger, build_holder
import matplotlib

matplotlib.use('Agg')

n_classes = {
    'cifar100': 100,
    'cifar10': 10,
    'mini_imagenet': 100,
    'imagenet100': 100,
}


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def parameter_setting():
    # mainly reference from https://github.com/RaptorMai/online-continual-learning
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Online Continual Learning PyTorch")
    ########################General#########################
    parser.add_argument('--seed', dest='seed', default=0, type=int,
                        help='Random seed, if seed < 0, it will not be set')

    parser.add_argument('--data', dest='data', default="mini_imagenet", type=str,
                        choices=['cifar10', 'cifar100', 'mini_imagenet', 'imagenet100'],
                        help='Path to the dataset. (default: %(default)s)')

    parser.add_argument('--run_time', dest='run_time', default=1, type=int,
                        help='the time of running')

    parser.add_argument('--agent', dest='agent', default="er", type=str,
                        choices=["er", "scr"],
                        help='agent')

    parser.add_argument('--retrieve', dest='retrieve', default="random", type=str,
                        choices=["random", "mfd", 'pearson', 'mir'],
                        help='retrieve method')

    parser.add_argument('--subsample', dest='subsample', default=50, type=int,
                        help='the size of subsample for calculating the pearson')

    parser.add_argument('--data_aug', dest='data_aug', default=True, type=boolean_string,
                        help='whether use data augmentation')

    parser.add_argument('--bf', dest='bf', default=True, type=boolean_string,
                        help='whether use BatchFormer')

    parser.add_argument('--dist', dest='dist', default=False, type=boolean_string,
                        help='whether use distlinear classifier')

    parser.add_argument('--drop_rate', dest='drop_rate', default=0.25, type=float,
                        help='the rate of drop out in BatchFormer')

    ########################Model#########################
    parser.add_argument('--vit', dest='vit', default=False, type=boolean_string, help='whether use ViT')

    ########################temperature for scr and normalized classifier#########################
    parser.add_argument('--temperature', dest='temperature', default=0.1, type=float,
                        help='the temperature of supervised contrastive learning')

    ########################certain-proned filter strategy#########################
    parser.add_argument('--certain_filter', dest='certain_filter', default=False, type=boolean_string,
                        help='whether use filter strategy for incoming batch')

    parser.add_argument('--filter_keep', dest='filter_keep', default=5, type=int,
                        help='Batch size for filter strategy (default: %(default)s)')

    ########################Revised Focal Loss#########################
    # default: rfl_alpha=0.25,rfl_sigma=0.5,rfl_miu=0.3
    parser.add_argument('--rfl_alpha', dest='rfl_alpha', default=0.25, type=float,
                        help='Revised Focal loss alpha')

    parser.add_argument('--rfl_sigma', dest='rfl_sigma', default=0.5, type=float,
                        help='Revised Focal loss sigma')

    parser.add_argument('--rfl_miu', dest='rfl_miu', default=0.3, type=float,
                        help='Revised Focal loss miu')

    ########################virtual knowledge distillation#########################
    parser.add_argument('--pfkd', dest='pfkd', default=True, type=boolean_string,
                        help='whethre use prototype feature knowledge distillation loss')

    parser.add_argument('--supcon_temperature', dest='supcon_temperature', default=0.1,
                        type=float,
                        help='supcon_temperature of feature distillation')

    parser.add_argument('--kd_trick', dest='kd_trick', default=False, type=boolean_string,
                        help='whethre use knowledge distillation loss')

    # kd_lamda * kd_loss
    parser.add_argument('--kd_lamda', dest='kd_lamda', default=2.0, type=float,
                        help='lamda used in knowledge distillation loss')

    parser.add_argument('--T', dest='T', default=2.0, type=float,
                        help='temperature for Distillation loss function, orginal paper set to 2, vkd is set to 20')

    # comparison between vkd and ckd, ckd is common knowledge distillation which is used in most CL scenario
    parser.add_argument('--kd_type', dest='kd_type', default="ckd", type=str,
                        choices=["ckd", "fkd", "dist"],
                        help='which type of kd will be used')

    parser.add_argument('--old_cls', dest='old_cls', default=False, type=boolean_string,
                        help='whether distillation only for old classes')

    ########################Experience Replay#########################
    parser.add_argument('--loss', dest='loss', default="ce", type=str,
                        choices=["ce", "focal", "rfocal", "scl"],
                        help='selected loss')

    # the strategy of ncm is same as SCR
    parser.add_argument('--classify', dest='classify', default="max", type=str,
                        choices=["ncm", "max"],
                        help='selected classification')

    parser.add_argument('--review_trick', dest='review_trick', default=False, type=boolean_string,
                        help='whether use review trick')

    ########################Focal Loss#########################
    # default : alpha=0.25, gamma=2.0
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

    # note that if last batch <= test_batch, then it will be the size of last batch
    parser.add_argument('--test_batch', dest='test_batch', default=128,
                        type=int,
                        help='Test batch size (default: %(default)s)')

    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.001,
                        help='weight_decay')

    ########################Data#########################
    parser.add_argument('--num_tasks', dest='num_tasks', default=5,
                        type=int,
                        help='Number of tasks (default: %(default)s)')

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
    # check whether let seed go
    if args.seed >= 0:
        initial(args)

    # set initial total tasks
    if args.data == 'cifar10':
        args.num_tasks = 5
    elif args.data == 'mini_imagenet':
        args.num_tasks = 10
    elif args.data == 'cifar100':
        args.num_tasks = 10
    elif args.data == 'imagenet100':
        args.num_tasks = 10

    args.num_classes = n_classes[args.data]

    if args.agent == "er":
        experience_replay(args, holder, log)
    elif args.agent == "scr":
        supervised_contrastive_replay(args, holder, log)
    else:
        raise NotImplementedError(
            'agent not supported: {}'.format(args.agent))


if __name__ == "__main__":
    # all config

    args = parameter_setting()

    holder = build_holder(args)

    log = Logger(holder + "/running_result.log", stream=sys.stdout)
    sys.stdout = log
    sys.stderr = log

    multiple_run(args, holder, log)
