#训练分类器算法

#1.加载并规范化CIFAR10
import torch
import torchvision
import torchvision.transforms as transforms

'''
transforms.Compose：用于将多个数据预处理操作组合在一起。
transforms.ToTensor()：将 PIL 图像或 NumPy ndarray 转换为 PyTorch 张量，并将图像的像素值归一化到 [0, 1] 之间。
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))：
 --第一个参数是每个通道的均值(mean)用于标准化，第二个参数是标准差(std)，此处均值和标准差都设置为0.5。
 --标准化公式：output=(input−mean)/std
 --对图像进行标准化可以使训练更加稳定，加速收敛。
'''
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#设定每个训练或测试批次中包含的样本数量，即每次从数据集中加载的图像数量。
batch_size = 4

'''
torchvision.datasets.CIFAR10
  --root='./data'：CIFAR-10 数据集将下载到该目录。
  --train=True：表示加载训练集，train=False 则加载测试集。
  --download=True：如果数据集不存在，则自动下载。
  --transform=transform：应用已定义的数据预处理操作。

torch.utils.data.DataLoader
  --trainset：输入训练集。
  --batch_size=batch_size：指定批次大小为 4。
  --shuffle=True：在每个 epoch 开始时随机打乱数据顺序，有助于提高模型泛化能力。
  --num_workers=2：使用 2 个线程加载数据，以加速数据读取。
'''
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

'''
加载测试集，与训练集的加载过程相似，但设置 train=False 表示这是测试集，并且 shuffle=False，因为测试集的顺序不需要随机化。
'''
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

'''
这是 CIFAR-10 数据集中的10个类别，分别是飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车。这些类别将在模型的输出层中用来进行分类。
'''
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


#展示训练图像
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))



#2.定义卷积神经网络
'''
torch.nn: PyTorch 中用来构建神经网络的模块，包含了层、损失函数等。
torch.nn.functional: 该模块提供了许多函数，例如激活函数和一些操作，这些函数通常不需要定义参数。
'''
import torch.nn as nn
import torch.nn.functional as F

'''
class Net(nn.Module): 创建一个名为Net的神经网络类，继承自nn.Module，这是所有神经网络模型的基类。
__init__ 方法: 初始化网络的结构，定义各个层。
'''
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        self.conv1 = nn.Conv2d(3, 6, 5): 定义第一个卷积层。
			输入通道数为 3（对应 RGB 图像）。
			输出通道数为 6（生成 6 个特征图）。
			卷积核大小为 5x5。
		self.pool = nn.MaxPool2d(2, 2): 定义最大池化层。
			池化窗口大小为 2x2，步长为 2，这将减小特征图的尺寸。
		self.conv2 = nn.Conv2d(6, 16, 5): 定义第二个卷积层。
			输入通道数为 6（来自第一个卷积层的 6 个特征图）。
			输出通道数为 16。
			卷积核大小仍为 5x5。
		self.fc1 = nn.Linear(16 * 5 * 5, 120): 定义第一个全连接层。
			输入特征维度是16×5×5，这个值来自第二个卷积层的输出尺寸。
			输出维度是 120。
		self.fc2 = nn.Linear(120, 84): 定义第二个全连接层。
			输入维度为 120，输出维度为 84。
		self.fc3 = nn.Linear(84, 10): 定义第三个全连接层。
			输入维度为 84，输出维度为 10，表示 CIFAR-10 数据集的 10 个类别。
        '''
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    '''
    forward(self, x): 定义前向传播的方法，x 是输入数据。
		self.conv1(x): 通过第一个卷积层处理输入。
		F.relu(...): 使用 ReLU 激活函数对卷积层的输出进行激活。
		self.pool(...): 通过最大池化减少特征图的尺寸。
		对第二个卷积层同样进行上述操作。
		torch.flatten(x, 1): 平展除了 batch 维度以外的所有维度。这里假设输入的特征图维度为(N,C,H,W)，经过flatten 后变为(N,C×H×W)。
		通过两个全连接层并再次应用 ReLU 激活函数。
		最后，最后一层 self.fc3 输出 10 个类别的得分。
    '''
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()



#3.定义 Loss 函数和优化器
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



#4.训练网络
#range(2): 这意味着将数据集遍历 2 次，通常可以更多，训练的轮次（epoch）由具体任务决定。
#epoch: 代表当前训练的轮次。
for epoch in range(2):  # loop over the dataset multiple times

    #running_loss: 用于累计当前 epoch 中的损失值，以便在训练过程中进行统计。
    running_loss = 0.0
    #enumerate(trainloader, 0): 迭代 trainloader 中的每一个 mini-batch，同时获取索引 i。
	#trainloader: 是一个数据加载器，提供输入图像及其对应标签。
    for i, data in enumerate(trainloader, 0):
        #从 data 中提取输入图像（inputs）和相应的标签（labels）。
        inputs, labels = data
        #在进行反向传播之前，先将之前的梯度清零。这是因为在 PyTorch 中梯度是累积的，若不清零，可能会导致梯度出现错误。
        optimizer.zero_grad()

'''
#前向传播、计算损失、反向传播与优化
outputs = net(inputs): 进行前向传播，通过网络（net）计算模型的输出。
loss = criterion(outputs, labels): 计算损失，使用损失函数（如交叉熵、均方误差等）来评估模型输出与真实标签的差异。
loss.backward(): 进行反向传播，计算每个参数的梯度，这一步是关键，因为它使得优化器能够更新模型的参数。
optimizer.step(): 更新模型的参数，根据计算出的梯度调整权重。
'''
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
'''
#打印统计信息
running_loss += loss.item(): 将当前 mini-batch 的损失加到 running_loss 中，loss.item() 用于提取损失的数值。
if i % 2000 == 1999: 每处理 2000 个 mini-batch 就打印一次损失。
print(...): 输出当前的 epoch 数、当前的 mini-batch 索引，以及该段时间内的平均损失。
running_loss = 0.0: 重置 running_loss 为 0，以开始计算下一段的损失。
'''
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

#保存模型
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)



#5.在测试数据上测试网络
dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

#保存并重新加载模型
net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))

outputs = net(images)

#得到最高能量的指数
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))

#计算准确率
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


#类别准确率
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')















