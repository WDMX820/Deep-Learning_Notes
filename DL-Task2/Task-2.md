# Task-2：学习 Learning PyTorch

## 什么是 PyTorch？

PyTorch 是一个基于 Python 的科学计算包，主要有两大用途：

- NumPy 的替代品，以使用 GPU 和其他加速器的强大功能。
- 一个用于实现神经网络的自动微分库。

## 本教程的目标：

- 大致了解 PyTorch 的 Tensor 库和神经网络。
- 训练小型神经网络以对图像进行分类

## 教程1：Tensor - 张量

张量是一种特殊的数据结构，与数组非常相似 和矩阵。在 PyTorch 中，我们使用张量对输入进行编码，并且 模型的输出以及模型的参数。

张量类似于 NumPy 的 ndarray，不同之处在于张量可以在 GPU 或其他专用硬件来加速计算。如果你熟悉 ndarrays，你将 使用 Tensor API 得心应手。如果没有，请快速阅读 API 演练。

```python
import torch
import numpy as np
```

### 张量初始化

可以通过多种方式初始化张量。请看以下示例：

**直接来自数据**

可以直接从数据创建张量。数据类型是自动推断的。

```python
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
```

**从 NumPy 数组**

可以从 NumPy 数组创建张量（反之亦然 - 请参阅 [Bridge with NumPy](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#bridge-to-np-label)）。

```python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

**从另一个张量：**

新张量保留参数张量的属性（形状、数据类型），除非显式覆盖。

```python
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")
```

Out：

```
Ones Tensor:
 tensor([[1, 1],
        [1, 1]])

Random Tensor:
 tensor([[0.8823, 0.9150],
        [0.3829, 0.9593]])
```

**使用随机值或常量值：**

`shape`是张量维度的元组。在下面的函数中，它确定输出张量的维数。

```python
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```

Out：

```
Random Tensor:
 tensor([[0.3904, 0.6009, 0.2566],
        [0.7936, 0.9408, 0.1332]])

Ones Tensor:
 tensor([[1., 1., 1.],
        [1., 1., 1.]])

Zeros Tensor:
 tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

### 张量属性

Tensor 属性描述其形状、数据类型和存储它们的设备。

```python
tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

Out：

```
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```

### 张量操作

超过 100 种张量操作，包括转置、索引、切片、 数学运算、线性代数、随机采样等 [此处](https://pytorch.org/docs/stable/torch.html)进行了全面描述。

它们中的每一个都可以在 GPU 上运行（通常比在 GPU 上更高的速度 CPU）。如果您使用的是 Colab，请转至 Edit > Notebook 分配 GPU 设置。

```python
# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  print(f"Device tensor is stored on: {tensor.device}")
```

Out：

```
Device tensor is stored on: cuda:0
```

尝试列表中的一些操作。 如果您熟悉 NumPy API，您会发现 Tensor API 使用起来轻而易举。

**标准的类似 numpy 的索引和切片：**

```python
tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)
```

Out：

```
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```

**联接张量**可用于沿给定维度连接一系列张量。 另请参见 [torch.stack](https://pytorch.org/docs/stable/generated/torch.stack.html)， 另一个加入 op 的张量与 .`torch.cat``torch.cat`

```python
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
```

Out：

```
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
```

**乘以张量**

```python
# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")
```

Out：

```
tensor.mul(tensor)
 tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

tensor * tensor
 tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
```

这将计算两个张量之间的矩阵乘法

```python
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")
```

Out：

```
tensor.matmul(tensor.T)
 tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])

tensor @ tensor.T
 tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])
```

**就地操作**具有后缀的操作就地操作。例如： ， ， 将更改 .`_``x.copy_(y)``x.t_()``x`

```python
print(tensor, "\n")
tensor.add_(5)
print(tensor)
```

Out：

```
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])
```

- **注意**

就地操作可以节省一些内存，但在计算导数时可能会产生问题，因为会立即丢失 历史。因此，不鼓励使用它们。

### 使用 NumPy 桥接

CPU 和 NumPy 数组上的张量可以共享其底层内存 locations 的 Locations，更改一个位置将更改另一个位置。

#### Tensor 到 NumPy 数组

```python
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
```

Out：

```
t: tensor([1., 1., 1., 1., 1.])
n: [1. 1. 1. 1. 1.]
```

张量的变化反映在 NumPy 数组中。

```python
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
```

Out：

```
t: tensor([2., 2., 2., 2., 2.])
n: [2. 2. 2. 2. 2.]
```

#### NumPy 数组到 Tensor

```python
n = np.ones(5)
t = torch.from_numpy(n)
```

NumPy 数组中的更改反映在张量中。

```python
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
```

Out：

```
t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
n: [2. 2. 2. 2. 2.]
```



## 教程2：torch.autograd

## 简要介绍`torch.autograd`

`torch.autograd`是 PyTorch 的自动微分引擎，它为 神经网络训练。在本节中，您将获得一个概念性的 了解 Autograd 如何帮助神经网络训练。

## 背景

神经网络 （NN） 是嵌套函数的集合，这些函数是 对一些输入数据执行。这些函数由参数（由权重和偏差组成）定义，在 PyTorch 中，这些*参数*存储在 张。

训练 NN 分为两个步骤：

**前向传播**：在 forward prop 中，NN 会做出最佳猜测 关于正确的输出。它通过其每个 函数进行猜测。

**反向传播**：在反向传播中，NN 调整其参数 与其猜测中的误差成正比。它通过遍历 从输出向后，使用 根据函数的参数 （*gradients*） 进行优化 使用 Gradient Descent 的参数。有关更详细的演练 的 backprop，请观看[此视频 3Blue1Brown](https://www.youtube.com/watch?v=tIeHLnjs5U8).

## 在 PyTorch 中的使用

让我们看一下单个训练步骤。 在此示例中，我们从 加载预训练的 resnet18 模型。 我们创建一个随机数据张量来表示具有 3 个通道的单个图像，高度和宽度为 64， 及其相应的初始化为一些随机值。label 在预训练模型中具有 形状 （1,1000）。`torchvision``label`

注意：本教程仅适用于 CPU，不适用于 GPU 设备（即使将张量移动到 CUDA）。

```python
import torch
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)
```

Out：

```
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /var/lib/ci-user/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth

  0%|          | 0.00/44.7M [00:00<?, ?B/s]
 37%|###6      | 16.4M/44.7M [00:00<00:00, 171MB/s]
 76%|#######5  | 33.8M/44.7M [00:00<00:00, 177MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 180MB/s]
```

接下来，我们通过模型运行输入数据，通过其每一层进行预测。 这是**前向传递**。

```python
prediction = model(data) # forward pass
```

我们使用模型的预测和相应的标签来计算误差 （）。 下一步是通过网络反向传播此错误。 当我们调用误差张量时，将启动向后传播。 然后，Autograd 会计算每个模型参数的梯度并将其存储在参数的属性中。`loss``.backward()``.grad`

```python
loss = (prediction - labels).sum()
loss.backward() # backward pass
```

接下来，我们加载一个优化器，在本例中为 SGD，学习率为 0.01，[动量](https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d)为 0.9。 我们在优化器中注册模型的所有参数。

```python
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
```

最后，我们调用 start gradient descent。优化器通过存储在 中的梯度来调整每个参数。`.step()``.grad`

```python
optim.step() #gradient descent
```

此时，您拥有训练神经网络所需的一切。

![image-20241009114554535](C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241009114554535.png)



## 教程3：神经网络

可以使用该包构建神经网络。`torch.nn`

现在，您已经大致了解了 ，依赖于定义模型并区分它们。 一个包含层和一个 返回 .`autograd``nn``autograd``nn.Module``forward(input)``output`

例如，看看这个对数字图像进行分类的网络：

![卷积网络](https://pytorch.org/tutorials/_images/mnist.png)

卷积网络



它是一个简单的前馈网络。它接受输入，提供 依次经过几层，最后给出 输出。

神经网络的典型训练过程如下：

- 定义具有一些可学习参数的神经网络（或 权重）
- 迭代输入数据集
- 通过网络处理输入
- 计算损失（输出距离正确还有多远）
- 将梯度传播回网络的参数
- 更新网络的权重，通常使用简单的更新规则：`weight = weight - learning_rate * gradient`

## 定义网络

让我们定义这个网络：

```python
'''
torch 是 PyTorch 库的核心模块，用于张量操作。
torch.nn 提供了构建神经网络所需的各种模块。
torch.nn.functional 包含了不需要状态的函数，例如激活函数和卷积操作。
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

#这里定义了一个名为 Net 的神经网络类，继承自 nn.Module。所有自定义的神经网络都需要继承这个类。
class Net(nn.Module):

    def __init__(self):
        #super(Net, self).__init__(): 调用父类的构造函数，初始化 nn.Module。
        super(Net, self).__init__()
        #卷积层
        #self.conv1: 输入通道为1（例如灰度图），输出通道为6，卷积核大小为5x5。 对应输出的特征图尺寸将是 (N, 6, 28, 28)。
        #self.conv2: 输入通道为6，输出通道为16，卷积核大小也为5x5。对应输出特征图尺寸为 (N, 16, 10, 10)。
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        #全连接层:
		#self.fc1: 输入尺寸为 16 * 5 * 5（来自第二个池化层的输出），输出120个特征。
		#self.fc2: 从120个特征到84个特征。
		#self.fc3: 从84个特征到10个输出（通常用于10个类别的分类任务）。
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
	#forward 方法定义了数据如何通过网络进行传播。
    def forward(self, input):
        '''
        卷积层 C1: 对输入进行卷积并使用 ReLU 激活函数。
		池化层 S2: 使用 2x2 最大池化层，对特征图尺寸进行减半，输出 (N, 6, 14, 14)。
		卷积层 C3: 再次进行卷积和 ReLU 激活。
		池化层 S4: 进一步池化，得到 (N, 16, 5, 5)。
        '''
        c1 = F.relu(self.conv1(input))
        s2 = F.max_pool2d(c1, (2, 2))
        c3 = F.relu(self.conv2(s2))
        s4 = F.max_pool2d(c3, 2)
        #将多维的特征图展平为一维张量，尺寸为 (N, 400)。
        s4 = torch.flatten(s4, 1)
        #f5: 将展平后的张量通过第一个全连接层，并使用 ReLU 激活。
		#f6: 通过第二个全连接层，并使用 ReLU 激活。
		#output: 最后输出层，将结果映射到10个类别。
        f5 = F.relu(self.fc1(s4))
        f6 = F.relu(self.fc2(f5))
        output = self.fc3(f6)
        return output
#创建 Net 类的实例，并打印其结构，以展示网络中包含的层及其参数。
net = Net()
print(net)
```

Out：

```
Net(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=120, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
```

**卷积和池化的解释：**

![image-20241009154528010](C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241009154528010.png)



你只需要定义函数，函数（计算梯度的地方）就会自动为你定义调用。 您可以在函数中使用任何 Tensor 操作。`forward``backward``autograd``forward`

模型的可学习参数由`net.parameters()`

```python
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
```

Out：

```
10
torch.Size([6, 1, 5, 5])
```



让我们尝试一个随机的 32x32 输入。 注意：此网络 （LeNet） 的预期输入大小为 32x32。要使用此网络 MNIST 数据集，请将数据集中的图像大小调整为 32x32。

```python
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
```

Out：

```
tensor([[ 0.1453, -0.0590, -0.0065,  0.0905,  0.0146, -0.0805, -0.1211, -0.0394,
         -0.0181, -0.0136]], grad_fn=<AddmmBackward0>)
```



使用 random 将所有参数和反向传播的 gradient 缓冲区归零 梯度：

```python
net.zero_grad()
out.backward(torch.randn(1, 10))
```



##### **注意：torch.nn`仅支持 mini-batch。整个包仅支持小批量样本的输入，而不支持 单个样本。`torch.nn**

例如，将采用 的 4D 张量。`nn.Conv2d``nSamples x nChannels x Height x Width`

如果您有单个样本，只需用于添加 虚构的批次维度。`input.unsqueeze(0)`

在继续之前，让我们回顾一下到目前为止您看到的所有类。

- **回顾：**

  `torch.Tensor`- 支持 autograd 的*多维数组* 操作如 .还*保持梯度* w.r.t. 张肌。`backward()``nn.Module`- 神经网络模块。*便捷的方式 封装参数*，以及用于将它们移动到 GPU 的帮助程序， 导出、加载等`nn.Parameter`- 一种 Tensor*，即 在作为属性分配给 .*`Module``autograd.Function`- 实现*前向和后向定义 的 autograd 操作*。每个操作都会在 至少一个连接到函数的节点 创建了一个 A 并*对其历史记录进行了编码*。`Tensor``Function``Tensor`

- **在这一点上，我们介绍了：**

  定义神经网络处理输入和向后调用

- **仍然留下：**

  计算损失

  更新网络的权重

## 损失函数

损失函数采用 （output， target） 对输入，并计算一个 估计输出与目标的距离的值。

在 nn 包。 一个简单的损失是：它计算均方误差 在输出和目标之间。`nn.MSELoss`

例如：

```python
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
```

Out：

```
tensor(1.3619, grad_fn=<MseLossBackward0>)
```



现在，如果你使用它的 attribute 向后走，你会看到一个计算图，看起来 喜欢这个：`loss``.grad_fn`

```
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> flatten -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
```



因此，当我们调用 时，整个图是微分的 w.r.t. 神经网络参数，图中所有具有 Tensor 的 Tensor 都将与 梯度。`loss.backward()``requires_grad=True``.grad`

为了说明这一点，让我们倒退几个步骤：

```python
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
```

Out：

```
<MseLossBackward0 object at 0x7fcf2d32ad10>
<AddmmBackward0 object at 0x7fcf2d32bc10>
<AccumulateGrad object at 0x7fcf2d32b910>
```



## 反向传播

要反向传播错误，我们所要做的就是 。 不过，您需要清除现有的渐变，否则渐变将是 累积到现有梯度。`loss.backward()`

现在我们调用 ，看看 conv1 的偏差 向后 （backward） 之前和之后的渐变。`loss.backward()`

```python
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
```

Out：

```
conv1.bias.grad before backward
None
conv1.bias.grad after backward
tensor([ 0.0081, -0.0080, -0.0039,  0.0150,  0.0003, -0.0105])
```



现在，我们已经了解了如何使用损失函数。

**稍后阅读：**

> 神经网络包包含各种模块和损失函数 它们构成了深度神经网络的构建块。包含 文档[在这里。](https://pytorch.org/docs/nn)

**唯一需要学习的是：**

> - 更新网络的权重

## 更新权重

实践中使用的最简单的更新规则是随机梯度 血统 （SGD）：

```python
weight = weight - learning_rate * gradient
```



我们可以使用简单的 Python 代码来实现这一点：

```python
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
```



但是，当您使用神经网络时，您希望使用各种不同的 更新 SGD、Nesterov-SGD、Adam、RMSProp 等规则。 为了实现这一点，我们构建了一个小包：即 实现所有这些方法。使用起来非常简单：`torch.optim`

```
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
```

注意

观察如何使用 .这是因为梯度是累积的 如 [Backprop](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#backprop) 部分所述。`optimizer.zero_grad()`



## 教程4：训练分类器

就是这样。您已经了解了如何定义神经网络、计算损失和生成 对网络权重的更新。

现在你可能会想，

## 数据呢？

通常，当您必须处理图像、文本、音频或视频数据时， 您可以使用将数据加载到 NumPy 数组中的标准 Python 包。 然后您可以将此数组转换为 .`torch.*Tensor`

- 对于图像，Pillow、OpenCV 等包很有用
- 对于音频，scipy 和 librosa 等软件包
- 对于文本，基于原始 Python 或 Cython 的加载，或者 NLTK 和 SpaCy 很有用

专门针对视觉，我们创建了一个名为 的包，其中包含用于常见数据集的数据加载器，例如 ImageNet、CIFAR10、MNIST 等以及用于图像的数据转换器，即 和 .`torchvision``torchvision.datasets``torch.utils.data.DataLoader`

这提供了极大的便利，并避免了编写样板代码。

在本教程中，我们将使用 CIFAR10 数据集。 它有类： 'airplane'， 'automobile'， 'bird'， 'cat'， 'deer'， '狗'， '青蛙'， '马'， '船'， '卡车'。CIFAR-10 中的图像是 大小 3x32x32，即大小为 32x32 像素的 3 通道彩色图像。

![cifar10](https://pytorch.org/tutorials/_images/cifar10.png)

CIFAR10

## 训练图像分类器

我们将按顺序执行以下步骤：

1. 使用CIFAR10`torchvision`
2. 定义卷积神经网络
3. 定义损失函数
4. 在训练数据上训练网络
5. 在测试数据上测试网络

### 1. 加载并规范化CIFAR10

使用 ，加载CIFAR10非常容易。`torchvision`

```python
import torch
import torchvision
import torchvision.transforms as transforms
```

torchvision 数据集的输出是范围 [0， 1] 的 PILImage 图像。 我们将它们转换为归一化范围 [-1， 1] 的 Tensor。

注意

如果在 Windows 上运行并收到 BrokenPipeError，请尝试将 torch.utils.data.DataLoader（） 的num_worker为 0。

```python
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
```

Out：

```
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz

  0%|          | 0/170498071 [00:00<?, ?it/s]
  0%|          | 98304/170498071 [00:00<03:09, 900190.29it/s]
  0%|          | 819200/170498071 [00:00<00:39, 4294821.94it/s]
  2%|1         | 3342336/170498071 [00:00<00:12, 13393272.26it/s]
  4%|3         | 6029312/170498071 [00:00<00:08, 18462090.94it/s]
  6%|5         | 9732096/170498071 [00:00<00:06, 24863381.60it/s]
  7%|7         | 12713984/170498071 [00:00<00:05, 26361300.24it/s]
  9%|9         | 15892480/170498071 [00:00<00:05, 27919752.82it/s]
 11%|#1        | 19333120/170498071 [00:00<00:05, 29775456.85it/s]
 14%|#3        | 23101440/170498071 [00:00<00:04, 32057679.66it/s]
 15%|#5        | 26345472/170498071 [00:01<00:04, 31355940.79it/s]
 17%|#7        | 29491200/170498071 [00:01<00:04, 30693723.47it/s]
 20%|#9        | 33390592/170498071 [00:01<00:04, 33106583.63it/s]
 22%|##1       | 36732928/170498071 [00:01<00:04, 32845367.31it/s]
 24%|##4       | 41746432/170498071 [00:01<00:03, 37916853.84it/s]
 28%|##8       | 48168960/170498071 [00:01<00:02, 45695209.39it/s]
 32%|###2      | 54951936/170498071 [00:01<00:02, 52192496.60it/s]
 36%|###6      | 61964288/170498071 [00:01<00:01, 57496529.21it/s]
 41%|####      | 69107712/170498071 [00:01<00:01, 61528436.14it/s]
 45%|####4     | 76087296/170498071 [00:01<00:01, 63984604.13it/s]
 49%|####9     | 83951616/170498071 [00:02<00:01, 68343463.60it/s]
 53%|#####3    | 90800128/170498071 [00:02<00:01, 67270465.26it/s]
 58%|#####7    | 98467840/170498071 [00:02<00:01, 70034312.61it/s]
 62%|######1   | 105512960/170498071 [00:02<00:00, 69392901.87it/s]
 66%|######6   | 113213440/170498071 [00:02<00:00, 71636708.65it/s]
 71%|#######   | 120389632/170498071 [00:02<00:00, 69268598.21it/s]
 75%|#######5  | 127893504/170498071 [00:02<00:00, 70922764.87it/s]
 79%|#######9  | 135036928/170498071 [00:02<00:00, 70212502.91it/s]
 84%|########3 | 142508032/170498071 [00:02<00:00, 71461090.73it/s]
 88%|########7 | 150011904/170498071 [00:02<00:00, 72463484.59it/s]
 92%|#########2| 157450240/170498071 [00:03<00:00, 73005422.30it/s]
 97%|#########6| 164790272/170498071 [00:03<00:00, 69440742.29it/s]
100%|##########| 170498071/170498071 [00:03<00:00, 51995818.15it/s]
Extracting ./data/cifar-10-python.tar.gz to ./data
Files already downloaded and verified
```



让我们展示一些训练图像，以便有趣。

```python
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
```

<img src="https://pytorch.org/tutorials/_images/sphx_glr_cifar10_tutorial_001.png" alt="cifar10 tutorial"  />

```
frog  plane deer  car
```



### 2. 定义卷积神经网络

从之前的 Neural Networks （神经网络） 部分复制神经网络，并将其修改为 拍摄 3 通道图像（而不是定义的 1 通道图像）。

```python
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
```

**注：卷积和池化操作通常是在全连接层之前的预处理操作**



### 3. 定义 Loss 函数和优化器

让我们使用 Classification Cross-Entropy 损失和带有动量的 SGD。

```python
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```



### 4. 训练网络

这是事情开始变得有趣的时候。 我们只需要遍历我们的数据迭代器，并将输入提供给 网络和优化。

```python
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
```

Out：

```
[1,  2000] loss: 2.144
[1,  4000] loss: 1.835
[1,  6000] loss: 1.677
[1,  8000] loss: 1.573
[1, 10000] loss: 1.526
[1, 12000] loss: 1.447
[2,  2000] loss: 1.405
[2,  4000] loss: 1.363
[2,  6000] loss: 1.341
[2,  8000] loss: 1.340
[2, 10000] loss: 1.315
[2, 12000] loss: 1.281
Finished Training
```

让我们快速保存经过训练的模型：

```python
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
```

有关保存 PyTorch 模型的更多详细信息，请参阅[此处](https://pytorch.org/docs/stable/notes/serialization.html)。



### 5. 在测试数据上测试网络

我们已经在训练数据集上训练了网络 2 次。 但是我们需要检查网络是否学到了任何东西。

我们将通过预测神经网络的类标签来检查这一点 outputs，并根据 ground-truth 进行检查。如果预测为 correct，我们将样本添加到正确预测列表中。

好的，第一步。让我们显示测试集中的图像以熟悉。

```python
dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))
```

![cifar10 tutorial](https://pytorch.org/tutorials/_images/sphx_glr_cifar10_tutorial_002.png)

```
GroundTruth:  cat   ship  ship  plane
```



接下来，让我们重新加载已保存的模型（注意：保存并重新加载模型 在这里不是必需的，我们只是为了说明如何做到这一点）：

```python
net = Net()
net.load_state_dict(torch.load(PATH, weights_only=True))
```

Out：

```
<All keys matched successfully>
```



好，现在让我们看看神经网络是怎么看上面的这些例子的：

```
outputs = net(images)
```

输出是 10 个类的能量。 类的能量越高，网络就越大 认为该图像属于特定类。 那么，让我们得到最高能量的指数：

```python
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))
```

Out：

```
Predicted:  cat   ship  truck ship
```



结果似乎相当不错。

让我们看看网络在整个数据集上的表现如何。此处的准确率为所有类别的平均值

```python
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
```

Out：

```
Accuracy of the network on the 10000 test images: 54 %
```



这看起来比偶然性要好得多，因为偶然性是 10% 的准确率（随机选择 10 个类中的一个类）。 似乎网络学到了一些东西。

嗯，表现良好的课程有哪些，表现良好的课程有哪些 表现不佳：

```python
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
```

Out：

```
Accuracy for class: plane is 37.9 %
Accuracy for class: car   is 62.2 %
Accuracy for class: bird  is 45.6 %
Accuracy for class: cat   is 29.2 %
Accuracy for class: deer  is 50.3 %
Accuracy for class: dog   is 45.9 %
Accuracy for class: frog  is 60.1 %
Accuracy for class: horse is 70.3 %
Accuracy for class: ship  is 82.9 %
Accuracy for class: truck is 63.1 %
```

#### 训练分类器的流程逻辑图：

​                               开始

​                                 │
​                                ▼

┌────────────────────────┐
│                          导入库                             │
└────────────────────────┘
                                 │
                                ▼
┌────────────────────────┐
│     定义数据预处理操作                           │
│         1. 转成Tensor                                 │
│         2. 归一化处理                                  │
└────────────────────────┘
                                 │
                                ▼
┌────────────────────────┐
│     加载训练和测试数据集                       │
│  1. CIFAR-10（训练集: train=True）    │
│  2. CIFAR-10（测试集: train=False）   │
│  3. 使用DataLoader加载数据                │
└────────────────────────┘
                                │
                               ▼
┌────────────────────────┐
│     定义神经网络模型结构                       │
│         1. 卷积层                                          │
│         2. 最大池化层                                  │
│         3. 全连接层                                      │
└────────────────────────┘
                                 │
                                ▼
┌────────────────────────┐
│        设置损失函数和优化器                    │
│    1. 损失函数: 交叉熵损失                      │
│    2. 优化器: SGD                                     │
└────────────────────────┘
                                 │
                                ▼
┌────────────────────────┐
│          训练模型                                         │
│  1. 对数据集进行多轮训练                     │
│  2. 前向传播计算输出                             │
│  3. 计算损失                                             │
│  4. 反向传播和优化                                 │
└────────────────────────┘
                                 │
                                ▼
┌────────────────────────┐
│           保存模型                                        │
│     torch.save保存模型参数                    │
└────────────────────────┘
                                 │
                                ▼
┌────────────────────────┐
│          测试模型                                         │
│  1. 读取测试样本                                     │
│  2. 前向传播得到预测                             │
│  3. 计算准确率                                         │
└────────────────────────┘
                                    │
                                   ▼
┌────────────────────────┐
│         输出每类的准确率                          │
└────────────────────────┘
                                  │
                                 ▼
                               结束





好的，那么接下来呢？

我们如何在 GPU 上运行这些神经网络？

## 在 GPU 上训练

就像将 Tensor 传输到 GPU 上一样，您将神经 net 到 GPU 上。

让我们首先将我们的设备定义为第一个可见的 cuda 设备（如果有） CUDA 可用：

```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)
```

Out：

```
cuda:0
```

本节的其余部分假定这是一个 CUDA 设备。`device`

然后这些方法将递归遍历所有模块并将其 参数和缓冲区添加到 CUDA 张量中：

```python
net.to(device)
```

请记住，您必须在每个步骤中发送输入和目标 也添加到 GPU：

```python
inputs, labels = data[0].to(device), data[1].to(device)
```

为什么与 CPU 相比，我没有注意到 MASSIVE 加速？因为您的网络 真的很小。

**锻炼：**尝试增加网络的宽度（参数 2 中的 第一个 ，以及第二个的参数 1 – 它们需要是相同的数字），看看你得到的是什么样的加速。`nn.Conv2d``nn.Conv2d`

**实现的目标**：

- 从高层次了解 PyTorch 的 Tensor 库和神经网络。
- 训练小型神经网络以对图像进行分类



### torch.nn的代码从0开始 - 编写教程

不断的对代码运行的部分进行重构，简化重复代码的书写和快速运行

[torch.nn 到底是什么？— PyTorch 教程 2.4.0+cu121 文档](https://pytorch.org/tutorials/beginner/nn_tutorial.html)





## 使用 TensorBoard 可视化模型、数据和训练

![image-20241009174534986](C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241009174534986.png)

在 [60 分钟闪电战](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)中， 我们向您展示如何加载数据， 通过我们定义为 的子类 的模型馈送它， 在训练数据上训练此模型，并在测试数据上对其进行测试。 为了查看发生了什么，我们打印出一些统计数据作为模型 正在进行培训，以了解培训是否在进行。 但是，我们可以做得更好：PyTorch 与 TensorBoard，一种旨在可视化神经 网络训练运行。本教程说明了它的一些 功能，使用 [Fashion-MNIST 数据集](https://github.com/zalandoresearch/fashion-mnist)，该数据集可以使用 torchvision.datasets 读入 PyTorch。`nn.Module`

在本教程中，我们将学习如何：

> 1. 读入数据并使用适当的转换（与前面的教程几乎相同）。
> 2. 设置 TensorBoard。
> 3. 写入 TensorBoard。
> 4. 使用 TensorBoard 检查模型架构。
> 5. 使用 TensorBoard 创建我们在上一个教程中创建的可视化的交互式版本，使用更少的代码

具体来说，在第 #5 点，我们将看到：

> - 检查训练数据的几种方法
> - 如何在训练时跟踪模型的性能
> - 如何评估模型在训练后的性能。

我们将从与 [CIFAR-10 教程](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)中类似的样板代码开始：

```python
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
```



我们将在该教程中定义一个类似的模型架构，只使 为了说明图像现在是 一个通道而不是三个，28x28 而不是 32x32：

```python
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
```



我们将定义相同的 和 之前的内容：`optimizer``criterion`

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```



![image-20241009172603953](C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241009172603953.png)



## 1. TensorBoard 设置

现在，我们将设置 TensorBoard，从 导入并定义一个 ，用于将信息写入 TensorBoard 的关键对象。`tensorboard``torch.utils``SummaryWriter`

```python
#从 PyTorch 的 torch.utils.tensorboard 模块导入了 SummaryWriter。SummaryWriter 是用于将数据写入 TensorBoard 的工具，可以记录诸如损失值、准确率、学习率等训练过程中的信息
from torch.utils.tensorboard import SummaryWriter

'''
创建了一个 SummaryWriter 的实例，并指定了用于存储日志文件的目录：
'runs/fashion_mnist_experiment_1': 这是指定的日志目录，表示所有与“Fashion MNIST”实验相关的日志文件将存储在这个路径下。包括：
'runs': 主目录，通常用于存储所有实验的输出。
'fashion_mnist_experiment_1': 子目录，特定于该实验的名称，这种方式有助于组织和区分不同实验的结果。
'''
writer = SummaryWriter('runs/fashion_mnist_experiment_1')
```

请注意，仅此行会创建一个文件夹。`runs/fashion_mnist_experiment_1`



## 2. 写入 TensorBoard

现在，让我们将图像写入 TensorBoard（具体来说，就是网格）中 使用 [make_grid](https://pytorch.org/vision/stable/utils.html#torchvision.utils.make_grid)。

```python
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
```



### --在本地计算机上使用 TensorBoard

现在正在运行

```
tensorboard --logdir=runs
```

从命令行，然后导航到 [http://localhost:6006](http://localhost:6006/) 应该会显示以下内容。

![../_static/img/tensorboard_first_view.png](https://pytorch.org/tutorials/_static/img/tensorboard_first_view.png)

### --在 Google Colab 中使用 TensorBoard

- 在 Colab 环境中，你可以使用以下命令启动 TensorBoard：

```python
%load_ext tensorboard
%tensorboard --logdir runs
```

这将直接在 Colab 界面中嵌入 TensorBoard。

![image-20241009175846303](C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241009175846303.png)

现在您知道如何使用 TensorBoard！但是，此示例可能是 在 Jupyter Notebook 中完成 - TensorBoard 真正擅长的地方在于 创建交互式可视化。我们接下来将介绍其中之一， 以及本教程结束时的更多内容。



## 3. 使用 TensorBoard 检查模型

TensorBoard 的优势之一是它能够可视化复杂模型 结构。让我们可视化我们构建的模型。

```python
writer.add_graph(net, images)
writer.close()
```



现在，在刷新 TensorBoard 时，您应该会看到一个“Graphs”选项卡，该选项卡 如下所示：

![../_static/img/tensorboard_model_viz.png](https://pytorch.org/tutorials/_static/img/tensorboard_model_viz.png)

继续并双击“Net”以查看它展开，看到一个 构成模型的各个操作的详细视图。

![image-20241009180021652](C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241009180021652.png)

TensorBoard 有一个非常方便的功能，用于可视化高维 数据，例如低维空间中的图像数据;我们将介绍这一点 下一个。



## 4. 向 TensorBoard 添加“投影仪”

我们可以可视化 higher 的 lower 维度表示 通过 [add_embedding](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_embedding) 方法的尺寸数据

```python
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
```



现在，在 TensorBoard 的“Projector”选项卡中，您可以看到这 100 个 图像 - 每张图像都是 784 维的 - 投影成三个 维度空间。此外，这是交互式的：您可以单击 并拖动以旋转三维投影。最后，一对 使可视化效果更易于查看的提示：选择 “color： label” ，以及启用“夜间模式”，这将使 由于背景为白色，因此更容易看到图像：

![../_static/img/tensorboard_projector.png](https://pytorch.org/tutorials/_static/img/tensorboard_projector.png)

![image-20241009181935210](C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241009181935210.png)

现在我们已经彻底检查了我们的数据，让我们展示一下 TensorBoard 如何 可以让跟踪模型训练和评估更清晰，从 训练。



## 5. 使用 TensorBoard 跟踪模型训练

在前面的示例中，我们简单地*打印*了模型的 running loss 每 2000 次迭代。现在，我们将 Running Loss 记录为 TensorBoard 以及模型的预测视图 making 的`plot_classes_preds`

```python
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
```



最后，让我们使用相同的模型训练代码来训练模型，来自 前面的教程，但每 1000 次将结果写入 TensorBoard 批处理而不是打印到控制台;这是使用 [add_scalar](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_scalar) 函数完成的。

此外，在训练时，我们将生成一个图像，显示模型的 预测与其中包含的四张图像的实际结果 批。

```python
running_loss = 0.0
for epoch in range(1):  # loop over the dataset multiple times

    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:    # every 1000 mini-batches...

            # ...log the running loss
            writer.add_scalar('training loss',
                            running_loss / 1000,
                            epoch * len(trainloader) + i)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals',
                            plot_classes_preds(net, inputs, labels),
                            global_step=epoch * len(trainloader) + i)
            running_loss = 0.0
print('Finished Training')
```



您现在可以查看标量选项卡，查看绘制的运行损失 在 15,000 次训练迭代中：

![../_static/img/tensorboard_scalar_runs.png](https://pytorch.org/tutorials/_static/img/tensorboard_scalar_runs.png)

此外，我们可以查看模型对 在整个学习过程中的任意批次。查看 “Images” 选项卡并滚动 在 “Predictions vs. actuals” 可视化下查看此内容; 这向我们表明，例如，在仅仅 3000 次训练迭代之后， 该模型已经能够区分视觉上不同的 衬衫、运动鞋和外套等类，尽管它不像 在以后的训练中变得自信：

![../_static/img/tensorboard_images.png](https://pytorch.org/tutorials/_static/img/tensorboard_images.png)

在前面的教程中，我们查看了模型 受过训练;在这里，我们将使用 TensorBoard 来绘制精确率召回率 曲线（[这里有](https://www.scikit-yb.org/en/latest/api/classifier/prcurve.html)很好的解释） 对于每个类。



## 6. 使用 TensorBoard 评估经过训练的模型

```python
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
```



您现在将看到一个包含精确率召回率的 “PR Curves” 选项卡 曲线。去四处逛逛;您将在 某些类模型具有近 100% 的“曲线下面积”， 而在其他 S 上，这个区域较低：

![../_static/img/tensorboard_pr_curves.png](https://pytorch.org/tutorials/_static/img/tensorboard_pr_curves.png)

这就是 TensorBoard 和 PyTorch 与它集成的介绍。 当然，您可以在 Jupyter 中执行 TensorBoard 执行的所有操作 Notebook 的 Notebook 中，但使用 TensorBoard，您可以获得交互式的视觉效果 默认情况下。

































