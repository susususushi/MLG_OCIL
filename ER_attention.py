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
from matplotlib.font_manager import FontProperties
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
        # print(model.in_planes)
        # for params in model.parameters():
        #     print(params.requires_grad)
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
        all_test = data_continuum.test_data()  # 取出测试集
        test_loaders = setup_test_loader(all_test, args)

        seen_so_far = torch.LongTensor(size=(0,)).cuda()
        acc_list = []
        losses_batch = AverageMeter()
        losses_mem = AverageMeter()
        # old_class_label = torch.tensor([]).long().cuda()

        ##############################singe dataset CIL training stage##############################
        for task_id, (x_train, y_train, labels) in enumerate(data_continuum):

            print('==>>> task id: {},  {}, {},'.format(task_id, x_train.shape, y_train.shape))
            train_dataset = dataset_transform(x_train, y_train, transform=transforms_match[args.data])
            train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0,
                                      drop_last=True)

            # new_class_label = torch.tensor(labels).long().cuda()
            ##############################singe task training stage##############################
            model.train()
            for batch_id, batch_data in enumerate(train_loader):
                batch_x, batch_y = batch_data
                if torch.cuda.is_available():
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()

                # if args.certain_filter:
                #     batch_x, batch_y = sampler.certain_filter(model, batch_x, batch_y)

                # ER-ACE part
                present = batch_y.unique()
                seen_so_far = torch.cat([seen_so_far, present]).unique()

                if args.data_aug:
                    batch_x = torch.cat([batch_x, aug_transform(batch_x)])
                    batch_y = torch.cat([batch_y, batch_y])

                optimizer.zero_grad()

                feats = model.features(batch_x)
                bf_batch_y = batch_y.clone()
                if args.bf:  # apply batchformer
                    pre_feats = feats
                    # feats = model.BatchFormer(feats.unsqueeze(1)).squeeze(1)

                    times = FontProperties(fname='./Times/times.ttf', size=16)

                    attn_output, attn_weights = model.BatchFormer.self_attn(
                        feats.unsqueeze(1), feats.unsqueeze(1), feats.unsqueeze(1), need_weights=True
                    )

                    # 后续处理 feats
                    feats = attn_output.squeeze(1)
                    # 可视化 Attention Map
                    attn_map = attn_weights[0].cpu().detach().numpy()  # 取第一个 batch 和第一个头
                    # CIFAR-100 类标签列表
                    class_names = [
                        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                        'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                        'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
                        'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
                        'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
                        'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
                        'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy',
                        'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal',
                        'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
                        'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
                        'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
                    ]
                    # 使用 seaborn 生成热力图
                    plt.figure(figsize=(12, 10))
                    ax = sns.heatmap(attn_map, annot=False, cmap='viridis', cbar=True)

                    # 生成标签名称列表
                    x_labels = [class_names[label] for label in batch_y]
                    y_labels = [class_names[label] for label in batch_y]
                    # 设置 x 轴标签
                    ax.set_xticks(np.arange(len(x_labels)) + 0.5)
                    ax.set_xticklabels(x_labels, rotation=90, fontsize=14, fontproperties=times)

                    # 设置 y 轴标签
                    ax.set_yticks(np.arange(len(y_labels)) + 0.5)
                    ax.set_yticklabels(y_labels, rotation=360, fontsize=14, fontproperties=times)

                    plt.title('Attention Map for Transformer Encoder Layer', fontsize=16, fontproperties=times)
                    plt.show()
                    plt.savefig('./attention/batch_idx:{}.pdf'.format(batch_id), bbox_inches='tight')  # 保存图像到文件
                    feats = torch.cat([pre_feats, feats], dim=0)
                    bf_batch_y = torch.cat([batch_y, batch_y], dim=0)

                x_logit = model.logits(feats)

                # ER-ACE part
                mask = torch.zeros_like(x_logit)
                # unmask current classes
                mask[:, present] = 1
                # not unmask unseen classes for PCR, unmask unseen classes for ER-ACE
                # mask[:, seen_so_far.max():] = 1 

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
                    if args.bf:  # apply batchformer
                        pre_mem_feats = mem_feats
                        mem_feats = model.BatchFormer(mem_feats.unsqueeze(1)).squeeze(1)
                        mem_feats = torch.cat([pre_mem_feats, mem_feats], dim=0)
                        bf_mem_y = torch.cat([mem_y, mem_y], dim=0)

                    mem_x_logit = model.logits(mem_feats)

                    # PCR part, compared with ER-ACE, it masks unseen classes
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
                        # mem_loss = mem_loss + args.kd_lamda * pfkd_crit(mem_feats, bf_mem_y)
                        mem_loss = mem_loss + args.kd_lamda * pfkd_crit(mem_x, mem_y, model)

                    mem_loss.backward()
                    # not .step() here, call .step() outside of if statement
                    losses_mem.update(mem_loss.item(), bf_mem_y.size(0))

                # the gradient have accumulated, just call .step()
                optimizer.step()

                batch_x, batch_y = batch_data
                sampler.update(batch_x, batch_y)
                if batch_id % 100 == 1:
                    print('==>>> it: {}, avg loss: {:.6f}, avg mem loss: {:.6f}'
                          .format(batch_id, losses_batch.avg(), losses_mem.avg()))
                    # break

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
                            #                            x_logit = model(batch_x)

                            feats = model.features(batch_x)
                            if args.bf:  # apply batchformer
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
                # teacher_model = deepcopy(model)
                # teacher_model.eval()
                # 这里更新出问题了之前是直接对teacher model赋值 导致结果一直没提升
                kd_criterion.update_teacher(model)
                # old_class_label = torch.cat([old_class_label, new_class_label])

            if args.pfkd:
                sampler.accumulate_update_prototype(model)  # 累积更新类原型
                pfkd_crit.update(sampler.mean_feat, sampler.mean_feat_label)

            ##############################evaluate stage##############################
            task_acc = evaluate(model, test_loaders, sampler, args)

            # save acc result
            acc_list.append(task_acc)

        # add the result of each run time to acc_list_all
        acc_list_all.append(np.array(acc_list))

        # torch.save(model.state_dict(), holder + '/net_params_run_{}.pth'.format(str(run_time).zfill(2)))
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
