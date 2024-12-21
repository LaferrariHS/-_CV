import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

num_classes = 10
num_examples_pre_epoch_for_train = 50000
num_examples_pre_epoch_for_eval = 10000

# 定义数据增强和预处理的转换操作
def get_transforms(distorted=True):
    if distorted:
        return transforms.Compose([
            transforms.RandomCrop(24),  # 随机裁剪到 24x24
            transforms.RandomHorizontalFlip(),  # 随机左右翻转
            transforms.ColorJitter(brightness=0.8, contrast=(0.2, 1.8)),  # 随机亮度和对比度调整
            transforms.ToTensor(),  # 转换为 Tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
        ])
    else:
        return transforms.Compose([
            transforms.Resize(32),  # 调整为 32x32 以匹配 CIFAR-10 原始图像尺寸
            transforms.CenterCrop(24),  # 从中心裁剪到 24x24
            transforms.ToTensor(),  # 转换为 Tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
        ])

# 数据加载函数
def get_data_loaders(data_dir, batch_size, distorted=True):
    # 使用 PyTorch 的 CIFAR10 数据集
    train_transform = get_transforms(distorted=distorted)
    test_transform = get_transforms(distorted=False)

    # 训练数据集
    train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)

    # 测试数据集
    test_dataset = CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    return train_loader, test_loader


