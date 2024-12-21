import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import cv2
import utils
import os

class AlexNet(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), output_shape=2):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=48, kernel_size=11, stride=4)
        self.bn1 = nn.BatchNorm2d(48)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(48, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(128, 192, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(192, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, output_shape)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv4(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv5(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.pool3(x)

        x = x.view(x.size(0), -1)  # 扁平化
        x = self.fc1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, lines, transform=None):
        self.lines = lines
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        name, label = self.lines[idx].strip().split(';')
        img_path = os.path.join(".\\data\\image\\train", name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

        # 确保图像类型为 float32
        img = img.astype(np.float32)
        
        if self.transform:
            img = self.transform(img)

        label = int(label)
        return img, label

if __name__ == "__main__":
    log_dir = "./logs/"
    batch_size = 128
    epochs = 50

    with open(r".\data\dataset.txt", "r") as f:
        lines = f.readlines()

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    # 数据增强和预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])

    # 创建数据集
    train_dataset = CustomDataset(lines[:num_train], transform=transform)
    val_dataset = CustomDataset(lines[num_train:], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = AlexNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 损失和准确率的保存
    best_val_loss = float('inf')
    patience = 10
    counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        running_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

    # 保存最后的权重
    torch.save(model.state_dict(), os.path.join(log_dir, 'last_model.pth'))
