import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,], [1,])])
 
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader

# 定义神经网络架构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x
    
# 定义训练模型
class LinearModel:
    # 定义初始化函数
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimist = self.create_optimizer(optimist)
        
    # 定义损失函数
    def create_cost(self, cost):
        support_cost = {
            'cross_entropy': nn.CrossEntropyLoss(),
            'mse': nn.MSELoss()
        }
 
        return support_cost[cost]

    # 定义优化器
    def create_optimizer(self, optimist, **rests):
        # 创建优化器
        support_optim = {
            'sgd': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'adam': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'rmsp':optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return support_optim[optimist]

    # 定义训练函数
    def train(self, trainloader, epochs = 10):
        for epoch in range(epochs):
            running_loss = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data #获取输入数据和标签   
                
                self.optimist.zero_grad() #清空上一步的残余更新参数值
                
                # 正向传播计算输出值
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels) #计算损失
                
                # 反向传播计算参数更新值
                loss.backward()
                self.optimist.step()
                
                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1)*1./len(train_loader), running_loss / 100))
                    running_loss = 0.0
        print('Finished Training')
        
    # 定义评估函数
    def evaluate(self, testloader):
        print('Evaluating ...')
        correct = 0
        total = 0
        
        with torch.no_grad():  # no grad when test and predict
            for data in testloader:
                images, labels = data
 
                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
        
if __name__ == '__main__':
    net = Net()
    model = LinearModel(net, 'cross_entropy', 'adam')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)

