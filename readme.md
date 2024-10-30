## 1. Requirements

The experiments are conducted using the following hardware and software:

- Hardware: NVIDIA GeForce RTX 3090 GPUs.
- Software: Please refer to `requirements.txt` for the detailed package versions. Conda is highly recommended.

## 2. Datasets

- CIFAR10 & CIFAR100 will be downloaded during the first run and stored in `continuums/datasets/cifar10` & `continuums/datasets/cifar100`.
- Mini-ImageNet: Download from https://www.kaggle.com/whitemoon/miniimagenet/download , and place it in `continuums/datasets/mini_imagenet/`.
- Download the ImageNet dataset from [this link](http://www.image-net.org/) and follow [this](https://github.com/danielchyeh/ImageNet-100-Pytorch) for ImageNet-100 dataset generation. Put the dataset in the `continuums/datasets/imagenet100_data`.

## 3. Sample commands to train

### CIFAR-10

```
python general_main.py --agent er --loss ce --classify max --data cifar10 --eps_mem_batch 10 --mem_size 200 --data_aug True --bf True --drop_rate 0.5 --retrieve random --pfkd True --kd_lamda 2.0
```

### Cifar-100

```
python general_main.py --agent er --loss ce --classify max --data cifar100 --eps_mem_batch 10 --mem_size 1000 --data_aug True --bf True --drop_rate 0.25 --retrieve random --pfkd True --kd_lamda 0.75
```

### Mini-ImageNet

```
python general_main.py --agent er --loss ce --classify max --data mini_imagenet --eps_mem_batch 10 --mem_size 1000 --data_aug True --bf True --drop_rate 0.25 --retrieve random --pfkd True --kd_lamda 0.1
```

### ImageNet100

```
python general_main.py --agent er --loss ce --classify max --data imagenet100 --eps_mem_batch 10 --mem_size 5000 --data_aug True --bf True --drop_rate 0.25 --retrieve random --pfkd True --kd_lamda 0.1
```



## Acknowledge

The code is mainly reference from https://github.com/RaptorMai/online-continual-learning. Thanks to [RaptorMai](https://github.com/RaptorMai) for his contribution to the framework and the implementation of recent state-of-the-art methods. 