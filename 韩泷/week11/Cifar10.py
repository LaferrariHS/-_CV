import numpy as np
import time
import math
import Cifar10_data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import Cifar10_data



# 定义神经网络结构
class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * 6 * 6, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 6 * 6)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
if __name__ == '__main__':
    # 实例化模型，定义损失函数和优化器
    data_dir = './data'  # 数据存放的目录
    batch_size = 64
    distorted = True  # 是否进行数据增强

    train_loader, test_loader = Cifar10_data.get_data_loaders(data_dir, batch_size, distorted)
    print('训练集样本数：', len(train_loader.dataset))
    print('测试集样本数：', len(test_loader.dataset))

    # 定义超参数
    max_steps = 4000
    batch_size = 100
    num_examples_for_eval = 10000
    data_dir = "Cifar_data/cifar-10-batches-bin"
    print("开始训练模型")
    model = CIFAR10Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 训练模型
    model.train()
    for step in range(max_steps):
        start_time = time.time()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            duration = time.time() - start_time

            if step % 100 == 0:
                examples_per_sec = batch_size / duration
                sec_per_batch = float(duration)
                print("step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)" % (step, loss.item(), examples_per_sec, sec_per_batch))
                break  # 每100步只计算一次

    # 计算最终的正确率
    model.eval()
    true_count = 0
    total_sample_count = num_examples_for_eval

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            true_count += (predicted == labels).sum().item()

    # 打印正确率信息
    accuracy = (true_count / total_sample_count) * 100
    print("accuracy = %.3f%%" % accuracy)
