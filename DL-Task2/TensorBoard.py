#基于CIFAR-10分类器的代码学习TensorBoard

#样板代码
# imports
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))])

# datasets
trainset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=True,
    transform=transform)
testset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=False,
    transform=transform)

# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                        shuffle=True, num_workers=2)


testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                        shuffle=False, num_workers=2)

# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))





#定义一个类似的模型架构、损失函数和优化器
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)





#1.TensorBoard设置
#从 PyTorch 的 torch.utils.tensorboard 模块导入了 SummaryWriter。SummaryWriter 是用于将数据写入 TensorBoard 的工具，可以记录诸如损失值、准确率、学习率等训练过程中的信息
from torch.utils.tensorboard import SummaryWriter

'''
创建了一个 SummaryWriter 的实例，并指定了用于存储日志文件的目录：
'runs/fashion_mnist_experiment_1': 这是指定的日志目录，表示所有与“Fashion MNIST”实验相关的日志文件将存储在这个路径下。包括：
'runs': 主目录，通常用于存储所有实验的输出。
'fashion_mnist_experiment_1': 子目录，特定于该实验的名称，这种方式有助于组织和区分不同实验的结果。
'''
writer = SummaryWriter('runs/fashion_mnist_experiment_1')


#2.写入 TensorBoard
'''
#获取随机训练图像
iter(trainloader) 创建一个迭代器 dataiter，该迭代器可以逐批次地遍历 trainloader 中的数据。
next(dataiter) 通过迭代器获取一个批次的数据，这里返回的是一个元组 images 和 labels：
	images: 包含一批训练图像的数据（通常是一个张量）。
	labels: 对应于这些图像的标签（分类信息）。
'''
dataiter = iter(trainloader)
images, labels = next(dataiter)

'''
#创建图像网格
torchvision.utils.make_grid(images) 将一批图像组合成一个网格（grid）格式的图像，以便更好地进行可视化。这通常有助于查看多张图像的样貌，尤其是在使用 TensorBoard 时。
'''
img_grid = torchvision.utils.make_grid(images)

'''
#显示图像
matplotlib_imshow() 是一个自定义函数（假设已定义），用于使用 Matplotlib 显示图像。img_grid 是之前生成的网格图像，并且 one_channel=True 表示如果图像是单通道（灰度图），则显示为单通道图像。
'''
matplotlib_imshow(img_grid, one_channel=True)

'''
#写入 TensorBoard
writer.add_image('four_fashion_mnist_images', img_grid) 将生成的图像网格 img_grid 写入 TensorBoard。图像将被标识为'four_fashion_mnist_images'，以便能够在TensorBoard界面中方便地查看。
'''
writer.add_image('four_fashion_mnist_images', img_grid)


#3.使用TensorBoard
%load_ext tensorboard
%tensorboard --logdir runs


#4.使用 TensorBoard 检查模型
writer.add_graph(net, images)
writer.close()


#5.向TensorBoard添加“投影仪”
# helper function
def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

# select random images and their target indices
images, labels = select_n_random(trainset.data, trainset.targets)

# get the class labels for each image
class_labels = [classes[lab] for lab in labels]

# log embeddings
features = images.view(-1, 28 * 28)
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images.unsqueeze(1))
writer.close()


#6.使用 TensorBoard 跟踪模型训练
# helper functions

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


#7.使用 TensorBoard 评估经过训练的模型
# 1. gets the probability predictions in a test_size x num_classes Tensor
# 2. gets the preds in a test_size Tensor
# takes ~10 seconds to run
class_probs = []
class_label = []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        output = net(images)
        class_probs_batch = [F.softmax(el, dim=0) for el in output]

        class_probs.append(class_probs_batch)
        class_label.append(labels)

test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
test_label = torch.cat(class_label)

# helper function
def add_pr_curve_tensorboard(class_index, test_probs, test_label, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_truth = test_label == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_truth,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()

# plot all the pr curves
for i in range(len(classes)):
    add_pr_curve_tensorboard(i, test_probs, test_label)











