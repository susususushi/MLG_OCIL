import os
import torch
import pickle
import numpy as np
from continuums.data_utils import create_task_composition, load_task_with_labels
from continuums.dataset_base import DatasetBase
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder


class ImageNet100(ImageFolder):
    def __init__(self, root, train, transform):
        super().__init__(os.path.join(root, 'train' if train else 'val'), transform)
        self.transform_in = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ConvertImageDtype(torch.float32),
        ])

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        img = self.transform_in(img)
        label = torch.tensor(label)
        return img, label


class SplitImagenet100(DatasetBase):
    def __init__(self, params):
        dataset = 'imagenet100'
        num_tasks = params.num_tasks
        super(SplitImagenet100, self).__init__(dataset, num_tasks, params)

    def download_load(self):
        train_dir = './datasets/imagenet100_data/train.pkl'
        test_dir = './datasets/imagenet100_data/val.pkl'

        train = pickle.load(open(train_dir, 'rb'))
        # self.train_data = train['data'].reshape((100000, 64, 64, 3))
        self.train_data = train['data'].transpose(0, 2, 3, 1)
        self.train_label = train['target']

        test = pickle.load(open(test_dir, 'rb'))
        # self.test_data = test['data'].reshape((10000, 64, 64, 3))
        self.test_data = test['data'].transpose(0, 2, 3, 1)
        self.test_label = test['target']

        # train_dataset = ImageNet100(base_path, train=True, transform=transforms.ToTensor())
        # self.train_data = torch.stack([img for img, _ in train_dataset])
        # self.train_label = torch.stack([label for _, label in train_dataset])
        #
        # test_dataset = ImageNet100(base_path, train=False, transform=transforms.ToTensor())
        # self.test_data = torch.stack([img for img, _ in test_dataset])
        # self.test_label = torch.stack([label for _, label in test_dataset])

    def new_run(self, **kwargs):
        self.setup()
        return self.test_set

    def new_task(self, cur_task, **kwargs):
        labels = self.task_labels[cur_task]
        x_train, y_train = load_task_with_labels(self.train_data, self.train_label, labels)
        return x_train, y_train, labels

    def setup(self):
        self.task_labels = create_task_composition(class_nums=100, num_tasks=self.task_nums,
                                                   fixed_order=self.params.fix_order)
        self.test_set = []
        for labels in self.task_labels:
            x_test, y_test = load_task_with_labels(self.test_data, self.test_label, labels)
            self.test_set.append((x_test, y_test))
