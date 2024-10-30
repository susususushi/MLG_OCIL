# -*- coding: utf-8 -*-

import torch
import os
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import random
import sys
from continuums.data_utils import transforms_match, dataset_transform, setup_test_loader

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
# matplotlib.use('Agg')
from models.resnet import Reduced_ResNet18, ResNet18
from continuums.continuum import continuum
from buffer.Reservoir_Random import Reservoir_Random
from loss.kd_manager import kd_manager
from loss.feature_distillation_loss import Feature_KD
from copy import deepcopy
import argparse
import math
from torch.utils.data import TensorDataset, Dataset, DataLoader
from setup_elements import setup_architecture, setup_opt, setup_crit, setup_augment
from AverageMeter import AverageMeter
from ipdb import set_trace
import torch.nn.init as init
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale, \
    RandomVerticalFlip
from logger import Logger
from evaluate import evaluate
from metrics import compute_performance
import time


def Normalization(x, Max, Min):
    x = (x - Min) / (Max - Min);
    return x


def experience_replay(args, holder, log):
    sys.stdout = log
    data_continuum = continuum(args.data, args)
    acc_list_all = []

    print(args)
    start = time.time()
    for run_time in range(args.run_time):
        # clear last task cache
        torch.cuda.empty_cache()

        print(args.data + "_mem_size=" + str(args.mem_size) + "_run_time=" + str(run_time))

        model = setup_architecture(args)
        aug_transform = setup_augment(args)
        optimizer = setup_opt(args.optimizer, model, args.learning_rate, args.weight_decay)
        criterion = setup_crit(args)

        sampler = Reservoir_Random(args)

        if torch.cuda.is_available():
            model = model.cuda()

        # for kd trick
        kd_criterion = kd_manager(args)
        pfkd_crit = Feature_KD(args)

        # dataset initialization
        data_continuum.new_run()
        all_test = data_continuum.test_data()
        test_loaders = setup_test_loader(all_test, args)

        seen_so_far = torch.LongTensor(size=(0,)).cuda()
        acc_list = []
        losses_batch = AverageMeter()
        losses_mem = AverageMeter()

        ##############################singe dataset CIL training stage##############################
        for task_id, (x_train, y_train, labels) in enumerate(data_continuum):

            print('==>>> task id: {},  {}, {},'.format(task_id, x_train.shape, y_train.shape))
            train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[args.data])
            train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0,
                                      drop_last=True)

            ##############################singe task training stage##############################
            model.train()
            for batch_id, batch_data in enumerate(train_loader):
                batch_x, batch_y = batch_data
                if torch.cuda.is_available():
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()

                present = batch_y.unique()
                seen_so_far = torch.cat([seen_so_far, present]).unique()

                if args.data_aug:
                    batch_x = torch.cat([batch_x, aug_transform(batch_x)])
                    batch_y = torch.cat([batch_y, batch_y])

                optimizer.zero_grad()

                feats = model.features(batch_x)
                bf_batch_y = batch_y.clone()
                # apply BatchFormer
                if args.bf:
                    pre_feats = feats
                    feats = model.BatchFormer(feats.unsqueeze(1)).squeeze(1)
                    feats = torch.cat([pre_feats, feats], dim=0)
                    bf_batch_y = torch.cat([batch_y, batch_y], dim=0)

                x_logit = model.logits(feats)

                mask = torch.zeros_like(x_logit)
                # unmask current classes
                mask[:, present] = 1

                # do the mask operation, mask the seen class but not occur currently
                x_logit = x_logit.masked_fill(mask == 0, -1e9)

                loss = criterion(x_logit, bf_batch_y)

                #######################knowledge distillation#######################
                if args.kd_trick:
                    loss = loss + kd_criterion.get_loss(batch_x, batch_y, task_id, model)

                loss.backward()
                losses_batch.update(loss.item(), bf_batch_y.size(0))

                if not sampler.is_empty():
                    # retrieve from memory buffer
                    mem_x, mem_y = sampler.retrieve(args.eps_mem_batch, model)

                    if torch.cuda.is_available():
                        mem_x = mem_x.cuda()
                        mem_y = mem_y.cuda()

                    # apply data augmentation
                    if args.data_aug:
                        mem_x = torch.cat([mem_x, aug_transform(mem_x)])
                        mem_y = torch.cat([mem_y, mem_y])

                    mem_feats = model.features(mem_x)
                    bf_mem_y = mem_y.clone()
                    # BatchFormer module
                    if args.bf:
                        pre_mem_feats = mem_feats
                        mem_feats = model.BatchFormer(mem_feats.unsqueeze(1)).squeeze(1)
                        mem_feats = torch.cat([pre_mem_feats, mem_feats], dim=0)
                        bf_mem_y = torch.cat([mem_y, mem_y], dim=0)

                    mem_x_logit = model.logits(mem_feats)
                    bf_present = bf_mem_y.unique()
                    mask = torch.zeros_like(mem_x_logit)
                    # unmask current classes
                    mask[:, bf_present] = 1

                    # do the mask operation, mask the seen class but not occur currently
                    mem_x_logit = mem_x_logit.masked_fill(mask == 0, -1e9)

                    mem_loss = criterion(mem_x_logit, bf_mem_y)

                    #######################knowledge distillation#######################
                    if args.kd_trick:
                        mem_loss = mem_loss + kd_criterion.get_loss(mem_x, mem_y, task_id, model)

                    if args.pfkd:
                        mem_loss = mem_loss + args.kd_lamda * pfkd_crit(mem_x, mem_y, model)

                    mem_loss.backward()
                    # not .step() here, call .step() outside of if statement
                    losses_mem.update(mem_loss.item(), bf_mem_y.size(0))

                # the gradient have accumulated, call .step()
                optimizer.step()

                batch_x, batch_y = batch_data
                sampler.update(batch_x, batch_y)
                if batch_id % 100 == 1:
                    print('==>>> it: {}, avg loss: {:.6f}, avg mem loss: {:.6f}'
                          .format(batch_id, losses_batch.avg(), losses_mem.avg()))

            #############################after train, conduct review trick##############################
            if args.review_trick:
                mem_x = sampler.buffer_img[:sampler.cur_idx]
                mem_y = sampler.buffer_label[:sampler.cur_idx]
                if mem_x.size(0) > 0:
                    rv_dataset = TensorDataset(mem_x, mem_y)
                    rv_loader = DataLoader(rv_dataset, batch_size=args.batch, shuffle=True,
                                           num_workers=0, drop_last=False)
                    for ep in range(1):
                        for i, batch_data in enumerate(rv_loader):
                            batch_x, batch_y = batch_data

                            if torch.cuda.is_available():
                                batch_x = batch_x.cuda()
                                batch_y = batch_y.cuda()

                            feats = model.features(batch_x)
                            # apply BatchFormer
                            if args.bf:
                                pre_feats = feats
                                feats = model.BatchFormer(feats.unsqueeze(1)).squeeze(1)
                                feats = torch.cat([pre_feats, feats], dim=0)
                                batch_y = torch.cat([batch_y, batch_y], dim=0)

                            x_logit = model.logits(feats)
                            loss = criterion(x_logit, batch_y)

                            optimizer.zero_grad()
                            loss.backward()
                            # down gradient, the same as learning rate / 10.0
                            params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
                            grad = [p.grad.clone() / 10.0 for p in params]
                            for g, p in zip(grad, params):
                                p.grad.data.copy_(g)
                            optimizer.step()

            ##############################update old model##############################
            if args.kd_trick:
                kd_criterion.update_teacher(model)

            if args.pfkd:
                sampler.accumulate_update_prototype(model, task_id)
                pfkd_crit.update(sampler.mean_feat, sampler.mean_feat_label)
            else:
                sampler.accumulate_update_prototype(model, task_id)

            ##############################evaluate stage##############################
            task_acc = evaluate(model, test_loaders, sampler, args)

            # save acc result
            acc_list.append(task_acc)

        # add the result of each run time to acc_list_all
        acc_list_all.append(np.array(acc_list))

        ##############################print acc result##############################
        # print all results for each run
        print("\n----------run {} result-------------".format(run_time))
        for acc in acc_list:
            print(acc)
        print("last task avr acc: ", np.mean(acc_list[len(acc_list) - 1]))

        ##############################save acc result##############################
        txt = holder + "/run_time = %d" % (run_time) + ".txt"
        with open(txt, "w") as f:
            for acc in acc_list:
                f.write(str(list(acc)) + "\n")
            f.write("last task avr acc: %lf" % np.mean(acc_list[len(acc_list) - 1]) + "\n")

        ##############################save setting parameter##############################
        if not os.path.exists(holder + '/setting.txt'):
            argsDict = args.__dict__
            with open(holder + '/setting.txt', 'w') as f:
                f.writelines('------------------ start ------------------' + '\n')
                for eachArg, value in argsDict.items():
                    f.writelines(eachArg + ' : ' + str(value) + '\n')
                f.writelines('------------------- end -------------------')
        print("\n\n")

    ##############################calculate avr result after args.run_time running##############################
    end = time.time()
    avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(np.array(acc_list_all))
    with open(holder + '/avr_end_result.txt', 'w') as f:
        f.write('Total run (second):{}\n'.format(end - start))
        f.write('Avg_End_Acc:{}\n'.format(avg_end_acc))
        f.write('Avg_End_Fgt:{}\n'.format(avg_end_fgt))
        f.write('Avg_Acc:{}\n'.format(avg_acc))
        f.write('Avg_Bwtp:{}\n'.format(avg_bwtp))
        f.write('Avg_Fwt:{}\n'.format(avg_fwt))

    print('----------- final average result -----------'
          .format(avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt))
    print(' Avg_End_Acc {}\n Avg_End_Fgt {}\n Avg_Acc {}\n Avg_Bwtp {}\n Avg_Fwt {}\n'
          .format(avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt))
    return avg_end_acc
