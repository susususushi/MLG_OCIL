import torch
import os
import numpy as np
import torch.nn as nn
import random
from torchsummary import summary
from continuums.data_utils import transforms_match, dataset_transform, setup_test_loader
import matplotlib as mpl
import sys
import matplotlib.pyplot as plt
import seaborn as sns
# from continuums.cifar10 import CIFAR10
# from continuums.cifar100 import CIFAR100
# from continuums.mini_imagenet import Mini_ImageNet
from models.resnet import Reduced_ResNet18, SupConResNet
from continuums.continuum import continuum
from buffer.Reservoir_Random import Reservoir_Random
from loss.SCRLoss import SupConLoss
import argparse
from torch.utils.data import Dataset, DataLoader, TensorDataset
from setup_elements import setup_architecture, setup_opt, setup_crit
from AverageMeter import AverageMeter
from ipdb import set_trace
from copy import deepcopy
import torch.nn.init as init
from logger import Logger
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
from evaluate import evaluate
from metrics import compute_performance
from setup_elements import input_size_match, feature_size_match

# Two methods to set GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# torch.cuda.set_device(0)
def copy_old_model(model):
    old_model = deepcopy(model).cuda()
    old_model.eval()
    # Freeze all parameters from the model, including the heads
    for param in old_model.parameters():
        param.requires_grad = False
    return old_model


def l2_norm(x, axit=1):  # add by czj
    norm = torch.norm(x, 2, axit, True)
    output = torch.div(x, norm)
    return output


def supervised_contrastive_replay(args, holder, log):
    sys.stdout = log

    data_continuum = continuum(args.data, args)
    acc_list_all = []
    print(args)
    for run_time in range(args.run_time):
        torch.cuda.empty_cache()
        print(args.data + "_mem_size=" + str(args.mem_size) + "_run_time=" + str(run_time))

        model = setup_architecture(args)
        optimizer = setup_opt(args.optimizer, model, args.learning_rate, args.weight_decay)
        criterion = setup_crit(args)
        if torch.cuda.is_available():
            model = model.cuda()

        sampler = Reservoir_Random(args)

        data_continuum.new_run()
        all_test = data_continuum.test_data()
        test_loaders = setup_test_loader(all_test, args)

        acc_list = []
        losses_batch = AverageMeter()
        aug_transform = nn.Sequential(
            RandomResizedCrop(size=(input_size_match[args.data][1], input_size_match[args.data][2]), scale=(0.2, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2)
        )

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
                combined_batch_x = batch_x
                combined_batch_y = batch_y

                if not sampler.is_empty():
                    mem_x, mem_y = sampler.retrieve(args.eps_mem_batch)
                    if torch.cuda.is_available():
                        mem_x = mem_x.cuda()
                        mem_y = mem_y.cuda()
                    combined_batch_x = torch.cat((combined_batch_x, mem_x))
                    combined_batch_y = torch.cat((combined_batch_y, mem_y))

                optimizer.zero_grad()
                combined_x_feat = model(combined_batch_x)
                # SCR part
                aug_batch_x = aug_transform(combined_batch_x)
                aug_x_feat = model(aug_batch_x)
                all_feat = torch.cat([combined_x_feat.unsqueeze(1), aug_x_feat.unsqueeze(1)], dim=1)

                loss_scr = criterion(all_feat, combined_batch_y)

                loss_scr.backward()
                optimizer.step()
                losses_batch.update(loss_scr.item(), combined_batch_y.size(0))

                sampler.update(batch_x, batch_y)
                if batch_id % 100 == 1:
                    print(
                        '==>>> it: {}, avg loss: {:.6f}'
                        .format(batch_id, losses_batch.avg()))
            #                            break

            ##############################after train review trick##############################
            if args.review_trick:
                mem_x = sampler.buffer_img[:sampler.cur_idx]
                mem_y = sampler.buffer_label[:sampler.cur_idx]
                if mem_x.size(0) > 0:
                    rv_dataset = TensorDataset(mem_x, mem_y)
                    rv_loader = DataLoader(rv_dataset, batch_size=args.eps_mem_batch, shuffle=True,
                                           num_workers=0, drop_last=True)
                    for ep in range(1):
                        for i, batch_data in enumerate(rv_loader):
                            # batch update
                            batch_x, batch_y = batch_data
                            if torch.cuda.is_available():
                                batch_x = batch_x.cuda()
                                batch_y = batch_y.cuda()

                            aug_batch_x = aug_transform(batch_x)

                            x_feat = model(batch_x)
                            aug_x_feat = model(aug_batch_x)

                            all_feat = torch.cat([x_feat.unsqueeze(1), aug_x_feat.unsqueeze(1)], dim=1)
                            loss_scr = criterion(all_feat, batch_y)
                            optimizer.zero_grad()
                            loss_scr.backward()

                            params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
                            grad = [p.grad.clone() / 10. for p in params]
                            for g, p in zip(grad, params):
                                p.grad.data.copy_(g)

                            optimizer.step()

                            ##############################evaluate stage##############################
            task_acc = evaluate(model, test_loaders, sampler, args)
            acc_list.append(np.array(task_acc))

        # add the result of each run time to acc_list_all
        acc_list_all.append(np.array(acc_list))
        ##############################print acc result##############################
        # print all results for each run
        for acc in acc_list:
            print(acc)
        print("last task avr acc: ", np.mean(acc_list[len(acc_list) - 1]))

        ##############################save acc result##############################
        txt = holder + "/run_time = %d" % (run_time) + ".txt"
        with open(txt, "w") as f:
            for acc in acc_list:
                f.write(str(acc) + "\n")
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
    avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt = compute_performance(np.array(acc_list_all))

    with open(holder + '/avr_end_result.txt', 'w') as f:
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
