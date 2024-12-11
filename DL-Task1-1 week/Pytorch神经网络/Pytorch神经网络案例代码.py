#数据集和数据加载器部分加载代码，并构建模型
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()


#超参数
learning_rate = 1e-3
batch_size = 64
epochs = 5


#损失函数
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()


#优化器
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


#全面实施
'''
train_loop 函数 - 训练深度学习模型
1. 参数
dataloader：一个数据加载器，可以批量加载训练数据。
model：正在训练的深度学习模型。
loss_fn：损失函数，用于计算模型的损失（错误）。
optimizer：优化器，用于更新模型的参数。
2. 主要步骤
获取数据集大小：size = len(dataloader.dataset)。
设置模型为训练模式：model.train()。这对一些层（如 dropout 和 batch normalization）影响其行为。
Training Loop：
使用 enumerate(dataloader) 迭代批处理数据，其中 X 是输入数据，y 是真实标签。
预测：模型通过输入 X 得到预测 pred。
计算损失：通过损失函数计算损失。
反向传播：loss.backward() 计算损失对于模型参数的梯度。
优化步骤：optimizer.step() 更新模型的参数。
清零梯度：optimizer.zero_grad() 清除梯度信息，为下一次迭代做准备。
输出当前损失信息：每 100 个批次输出一次当前的损失。
'''
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")   
            
'''
test_loop 函数 - 测试深度学习模型
1. 参数
dataloader：一个数据加载器，加载测试数据。
model：进行评估的深度学习模型。
loss_fn：损失函数，用于计算测试集的损失。
2. 主要步骤
设置模型为评估模式：model.eval()，确保测试时模型正确处理。
获取数据集大小和批次数。
初始化损失和正确计数：test_loss 和 correct 用于记录测试结果。
禁用梯度计算：with torch.no_grad()，这可以减少内存使用和加快计算。
测试 Loop：
遍历测试数据：通过 dataloader 获得输入 X 和真实标签 y。
预测：计算模型对输入 X 的预测 pred。
累计损失：累加测试损失。
计算准确率：通过比较预测结果与真实标签，计算分类正确的数量。
3. 输出结果
计算平均测试损失和准确率，并打印结果。
'''
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 初始化损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 设定训练轮数 (Epochs)
epochs = 10

# 循环训练和测试
for t in range(epochs):
    #使用 for 循环迭代每个 epoch。在每次循环中，输出当前 epoch 的编号，以便跟踪训练进度。
    print(f"Epoch {t+1}\n-------------------------------")
    #调用 train_loop 函数来执行一次完整的训练过程，使用提供的训练数据加载器 (train_dataloader)、模型 (model)、损失函数 (loss_fn) 和优化器 (optimizer)。
    train_loop(train_dataloader, model, loss_fn, optimizer)
    #在每个 epoch 结束后，调用 test_loop 函数来评估模型在测试集上的表现，使用提供的测试数据加载器 (test_dataloader) 和损失函数 (loss_fn)。
    test_loop(test_dataloader, model, loss_fn)
print("Done!")



