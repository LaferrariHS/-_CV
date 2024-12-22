import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

def conv2d_bn(in_channels, out_channels, kernel_size, stride=1, padding='same'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class InceptionModule(nn.Module):
    def __init__(self, in_channels):
        super(InceptionModule, self).__init__()
        self.branch1x1 = conv2d_bn(in_channels, 64, kernel_size=1)

        self.branch5x5 = nn.Sequential(
            conv2d_bn(in_channels, 48, kernel_size=1),
            conv2d_bn(48, 64, kernel_size=5)
        )

        self.branch3x3dbl = nn.Sequential(
            conv2d_bn(in_channels, 64, kernel_size=1),
            conv2d_bn(64, 96, kernel_size=3),
            conv2d_bn(96, 96, kernel_size=3)
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            conv2d_bn(in_channels, 32, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1x1(x)
        branch2 = self.branch5x5(x)
        branch3 = self.branch3x3dbl(x)
        branch4 = self.branch_pool(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

class InceptionV3(nn.Module):
    def __init__(self, num_classes=1000):
        super(InceptionV3, self).__init__()

        self.conv1 = conv2d_bn(3, 32, kernel_size=3, stride=2, padding='valid')
        self.conv2 = conv2d_bn(32, 32, kernel_size=3, padding='valid')
        self.conv3 = conv2d_bn(32, 64, kernel_size=3)

        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv4 = conv2d_bn(64, 80, kernel_size=1, padding='valid')
        self.conv5 = conv2d_bn(80, 192, kernel_size=3, padding='valid')
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.inception1 = nn.Sequential(InceptionModule(192), InceptionModule(256), InceptionModule(288))
        self.inception2 = nn.Sequential(InceptionModule(288), InceptionModule(768), InceptionModule(768))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool2(x)

        x = self.inception1(x)
        x = self.inception2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def preprocess_input(img_path):
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = Image.open(img_path)
    img = preprocess(img)
    img = img.unsqueeze(0)  # 增加一个维度
    return img

if __name__ == '__main__':
    model = InceptionV3()
    model.load_state_dict(torch.load("inception_v3_weights.pth"))  # 假设您有转换权重
    model.eval()

    img_path = 'elephant.jpg'
    img = preprocess_input(img_path)

    with torch.no_grad():
        preds = model(img)
    print('Predicted:', preds)
