# -*- coding: utf-8 -*-

import torch
import os
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import random
import sys
from torchsummary import summary
from continuums.data_utils import transforms_match, dataset_transform
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
from AverageMeter import AverageMeter
from ipdb import set_trace
import torch.nn.init as init
from torch.utils.data import TensorDataset, Dataset, DataLoader


def evaluate(model, test_loaders, sampler, args):
    task_acc = np.zeros((len(test_loaders),))
    if args.classify == "ncm":
        sampler.update_prototype(model)  # Update the current NCM classifier
    model.eval()
    for _, test_loader in enumerate(test_loaders):
        acc_x, acc_y = 0, 0

        for batch_id, batch_data in enumerate(test_loader):
            batch_x_test, batch_y_test = batch_data

            if torch.cuda.is_available():
                batch_x_test = batch_x_test.cuda()
                batch_y_test = batch_y_test.cuda()

            with torch.no_grad():
                pred = None
                if args.classify == "ncm":
                    x_test_feat = model.features(batch_x_test)
                    x_test_feat = nn.functional.normalize(x_test_feat, p=2, dim=1)
                    pred = sampler.classify_mat(x_test_feat)
                elif args.classify == "max":
                    x_test_logit = model(batch_x_test)
                    pred = torch.argmax(x_test_logit, dim=1)

                else:
                    raise NotImplementedError(
                        'classify method not supported: {}'.format(args.classify))

                correct_x = torch.eq(pred, batch_y_test).int().sum().item()
                correct_y = batch_y_test.size(0)

                acc_x += correct_x
                acc_y += correct_y

        task_acc[_] = 1.0 * acc_x / acc_y
    print(task_acc)

    return task_acc
