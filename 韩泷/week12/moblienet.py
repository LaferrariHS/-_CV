import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def relu6(x):
    return torch.clamp(x, min=0, max=6)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), strides=(1, 1)):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=strides, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DepthwiseConvBlock(nn.Module):
    def __init__(self, in_channels, pointwise_conv_filters, depth_multiplier=1, strides=(1, 1), block_id=1):
        super(DepthwiseConvBlock, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * depth_multiplier, kernel_size=(3, 3), stride=strides, padding='same', groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels * depth_multiplier),
            nn.ReLU(inplace=True)
        )

        self.pointwise_conv = nn.Sequential(
            nn.Conv2d(in_channels * depth_multiplier, pointwise_conv_filters, kernel_size=(1, 1), padding='same', bias=False),
            nn.BatchNorm2d(pointwise_conv_filters),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise_conv(x)
        return self.pointwise_conv(x)

class MobileNet(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), depth_multiplier=1, dropout=1e-3, num_classes=1000):
        super(MobileNet, self).__init__()

        self.model = nn.Sequential(
            ConvBlock(input_shape[0], 32),  # 224 x 224 x 3 -> 224 x 224 x 32
            DepthwiseConvBlock(32, 64, depth_multiplier=depth_multiplier, block_id=1),  # 224 x 224 x 32 -> 224 x 224 x 64
            DepthwiseConvBlock(64, 128, depth_multiplier=depth_multiplier, strides=(2, 2), block_id=2),  # -> 112 x 112 x 128
            DepthwiseConvBlock(128, 128, depth_multiplier=depth_multiplier, block_id=3),  # -> 112 x 112 x 128
            DepthwiseConvBlock(128, 256, depth_multiplier=depth_multiplier, strides=(2, 2), block_id=4),  # -> 56 x 56 x 256
            DepthwiseConvBlock(256, 256, depth_multiplier=depth_multiplier, block_id=5),  # -> 56 x 56 x 256
            DepthwiseConvBlock(256, 512, depth_multiplier=depth_multiplier, strides=(2, 2), block_id=6),  # -> 28 x 28 x 512
            *[DepthwiseConvBlock(512, 512, depth_multiplier=depth_multiplier, block_id=i) for i in range(7, 12)],  # 12个512的Depthwise Conv Block
            DepthwiseConvBlock(512, 1024, depth_multiplier=depth_multiplier, strides=(2, 2), block_id=12),  # -> 14 x 14 x 1024
            DepthwiseConvBlock(1024, 1024, depth_multiplier=depth_multiplier, block_id=13),  # -> 14 x 14 x 1024
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 7 x 7 x 1024 -> 1 x 1 x 1024
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        return self.fc(x)

def preprocess_input(img_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img = Image.open(img_path)
    img = preprocess(img)
    img = img.unsqueeze(0)  # 增加一个维度
    return img

if __name__ == '__main__':
    model = MobileNet()
    model.eval()  # 切换到评估模式

    img_path = 'elephant.jpg'
    img = preprocess_input(img_path)

    with torch.no_grad():
        preds = model(img)
    
    predicted_class = torch.argmax(preds).item()
    print('Predicted class:', predicted_class)
