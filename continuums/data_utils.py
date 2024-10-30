import numpy as np
import torch
from torch.utils import data
from torchvision import transforms
import os

# from utils.setup_elements import transforms_match

mean = {
    "cifar10": (0.4914, 0.4822, 0.4465),
    "cifar100": (0.5071, 0.4867, 0.4408),
    "mini_imagenet": (0.485, 0.456, 0.406),
    "imagenet100": (0.485, 0.456, 0.406),
}
std = {
    "cifar10": (0.2023, 0.1994, 0.2010),
    "cifar100": (0.2675, 0.2565, 0.2761),
    "mini_imagenet": (0.229, 0.224, 0.225),
    "imagenet100": (0.229, 0.224, 0.225),
}

transforms_match = {
    'cifar10': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'cifar100': transforms.Compose([
        transforms.ToTensor(),
        # uncomment the following line if using a VIT model
        # transforms.Resize(224),
    ]),
    'mini_imagenet': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'imagenet100': transforms.Compose([
        transforms.ToTensor(),
    ]),
}


def create_task_composition(class_nums, num_tasks, fixed_order=False):
    classes_per_task = class_nums // num_tasks

    total_classes = classes_per_task * num_tasks

    label_array = np.arange(0, total_classes)

    if not fixed_order:
        np.random.shuffle(label_array)

    task_labels = []
    for tt in range(num_tasks):
        tt_offset = tt * classes_per_task

        task_labels.append(list(label_array[tt_offset:tt_offset + classes_per_task]))

        print('Task: {}, Labels:{}'.format(tt, task_labels[tt]))
    return task_labels


def load_task_with_labels_torch(x, y, labels):
    tmp = []
    for i in labels:
        tmp.append((y == i).nonzero().view(-1))
    idx = torch.cat(tmp)
    return x[idx], y[idx]


def load_task_with_labels(x, y, labels):
    tmp = []
    for i in labels:
        tmp.append((np.where(y == i)[0]))

    idx = np.concatenate(tmp, axis=None)
    return x[idx], y[idx]


class dataset_transform(data.Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = torch.from_numpy(y).type(torch.LongTensor)
        self.transform = transform  # save the transform

    def __len__(self):
        return len(self.y)  # self.x.shape[0]  # return 1 as we have only one image

    def __getitem__(self, idx):
        # return the augmented image
        if self.transform:
            x = self.transform(self.x[idx])
        else:
            x = self.x[idx]

        return x.float(), self.y[idx]


def setup_test_loader(test_data, params):
    test_loaders = []

    for (x_test, y_test) in test_data:
        test_dataset = dataset_transform(x_test, y_test, transform=transforms_match[params.data])
        test_loader = data.DataLoader(test_dataset, batch_size=params.test_batch, shuffle=True, num_workers=0,
                                      drop_last=False)
        test_loaders.append(test_loader)
    return test_loaders


def shuffle_data(x, y):
    perm_inds = np.arange(0, x.shape[0])
    np.random.shuffle(perm_inds)
    rdm_x = x[perm_inds]
    rdm_y = y[perm_inds]
    return rdm_x, rdm_y
