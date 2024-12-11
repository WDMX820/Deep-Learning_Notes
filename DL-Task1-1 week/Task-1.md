# Task-1：学习 google colab

完成google colab基本教程的学习，从使用入门到机器学习示例（https://colab.research.google.com/）

## 什么是 Colab？

借助 Colaboratory（简称 Colab），您可在浏览器中编写和执行 Python 代码，并且：

- 无需任何配置
- 免费使用 GPU
- 轻松共享

## **使用入门**

您正在阅读的文档并非静态网页，而是一个允许您编写和执行代码的交互式环境，称为 **Colab 笔记本**。

例如，以下**代码单元格**包含一个简短的 Python 脚本，该脚本会计算值、将其存储在变量中并输出结果：

```python
seconds_in_a_day = 24 * 60 * 60
seconds_in_a_day
```

要执行上述单元格中的代码，请点击选择它，然后按代码左侧的“播放”按钮，或使用键盘快捷键“Command/Ctrl+Enter”。要修改代码，只需点击单元格，然后开始修改。

您在某个单元格中定义的变量之后可用在其他单元格中：

```python
seconds_in_a_week = 7 * seconds_in_a_day
seconds_in_a_week
```

对于 Colab 笔记本，您可以将<strong>可执行代码</strong>、<strong>富文本</strong>以及<strong>图像</strong>、<strong>HTML</strong>、<strong>LaTeX</strong> 等内容合入 1 个文档中。当您创建自己的 Colab 笔记本时，系统会将这些笔记本存储在您的 Google 云端硬盘账号名下。您可以轻松地将 Colab 笔记本共享给同事或好友，允许他们评论甚至修改笔记本。要了解详情，请参阅 <a href="/notebooks/basic_features_overview.ipynb">Colab 概览</a>。要创建新的 Colab 笔记本，您可以使用上方的“文件”菜单，也可以使用以下链接：<a href="http://colab.research.google.com#create=true">创建新的 Colab 笔记本</a>。

Colab 笔记本是由 Colab 托管的 Jupyter 笔记本。如需详细了解 Jupyter 项目，请访问 [jupyter.org](https://www.jupyter.org/)。

## 机器学习案例

### 案例  Text Classification with Movie Reviews

 此笔记本使用评论文本将电影评论分类为*正面*或*负面*。这是*二元*（或双类）分类的一个示例，这是一种重要且广泛适用的机器学习问题。

我们将使用 [IMDB 数据集](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb)，其中包含来自 [Internet Movie Database](https://www.imdb.com/) 的 50,000 条电影评论的文本。这些分为 25,000 条用于培训的评论和 25,000 条用于测试的评论。训练集和测试集是*平衡*的，这意味着它们包含相同数量的正面和负面评论。

此笔记本使用 [tf.keras](https://www.tensorflow.org/api_docs/python/tf/keras)（一种高级 API）在 TensorFlow 中构建和训练模型，以及 [TensorFlow Hub](https://www.tensorflow.org/hub)（一种用于迁移学习.有关使用 [``](https://www.tensorflow.org/api_docs/python/tf/keras)的更高级文本分类教程，请参阅 [MLCC 文本分类指南](https://developers.google.com/machine-learning/guides/text-classification/)。

#### 1、设置 - 导入相关库

```python
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
```

![image-20241008143700553](C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241008143700553.png)

#### 2、下载 IMDB 数据集

IMDB 数据集在 [TensorFlow 数据集](https://github.com/tensorflow/datasets)上可用。

以下代码将 IMDB 数据集下载到您的计算机（或 colab 运行时）：

```python
train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"], 
                                  batch_size=-1, as_supervised=True)

train_examples, train_labels = tfds.as_numpy(train_data)
test_examples, test_labels = tfds.as_numpy(test_data)
```

![image-20241008143611404](C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241008143611404.png)

#### 3、探索数据

让我们花点时间了解一下数据的格式。每个示例都是表示电影评论的句子和相应的标签。该句子未以任何方式进行预处理。标签是 0 或 1 的整数值，其中 0 是负面评论，1 是正面评论。

![image-20241008144253349](C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241008144253349.png)

让我们打印前 10 个示例。

![image-20241008144221666](C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241008144221666.png)

我们还打印前 10 个标签。

![image-20241008144428358](C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241008144428358.png)

#### 4、构建模型

神经网络是通过堆叠层创建的，这需要三个主要的架构决策：

- 如何表示文本？
- 在模型中使用多少层？
- 每层使用多少*个隐藏单位*？

在此示例中，输入数据由句子组成。要预测的标签为 0 或 1。

表示文本的一种方法是将句子转换为嵌入向量。我们可以使用预训练的文本嵌入作为第一层，这将有两个好处：

- 我们不必担心文本预处理，
- 我们可以从迁移学习中受益。

在此示例中，我们将使用 [TensorFlow Hub](https://www.tensorflow.org/hub) 中名为 [google/nnlm-en-dim50/2](https://tfhub.dev/google/nnlm-en-dim50/2) 的模型。

对于本教程，还有另外两个模型需要测试：

- [google/nnlm-en-dim50-with-normalization/2](https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2) - 与 [google/nnlm-en-dim50/2](https://tfhub.dev/google/nnlm-en-dim50/2) 相同，但具有额外的文本规范化以删除标点符号。这有助于更好地覆盖输入文本上标记的词汇内嵌入。
- [google/nnlm-en-dim128-with-normalization/2](https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2) - 嵌入维度为 128 的较大模型，而不是较小的 50 个模型。

我们首先创建一个使用 TensorFlow Hub 模型嵌入句子的 Keras 层，然后在几个输入示例中进行尝试。请注意，生成的嵌入的输出形状是预期的： 。`(num_examples, embedding_dimension)`

**之后内容详见：**[使用电影评论进行文本分类 |TensorFlow Hub](https://www.tensorflow.org/hub/tutorials/tf2_text_classification)







# Task-2：熟悉深度学习库pytorch

## 一、Learn the Basics

大多数机器学习工作流都涉及处理数据、创建模型、优化模型 参数，并保存经过训练的模型。本教程向您介绍完整的 ML 工作流 在 PyTorch 中实现，并提供了用于了解有关每个概念的更多信息的链接。

我们将使用 FashionMNIST 数据集来训练一个神经网络，该神经网络预测输入图像是否属于 到以下类别之一：T 恤/上衣、裤子、套头衫、连衣裙、外套、凉鞋、衬衫、运动鞋、 包或踝靴。

## 二、Tensors

张量是一种专门的数据结构，与数组和矩阵非常相似。 在 PyTorch 中，我们使用张量对模型的输入和输出以及模型的参数进行编码。

张量类似于 [NumPy 的](https://numpy.org/) ndarrays，不同之处在于张量可以在 GPU 或其他硬件加速器上运行。事实上，张量和 NumPy 数组通常可以共享相同的底层内存，无需复制数据（请参阅[使用 NumPy 的 Bridge](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#bridge-to-np-label)）。张 还针对自动微分进行了优化（我们将在后面的 [Autograd](https://pytorch.org/tutorials/beginner/basics/autograd_tutorial.html) 部分看到更多相关信息）。如果您熟悉 ndarrays，那么您将熟悉 Tensor API。如果没有，请继续关注！

## 1、初始化 Tensor

可以通过多种方式初始化张量。请看以下示例：

**直接来自数据**

可以直接从数据创建张量。数据类型是自动推断的。

```python
data = [[1, 2],[3, 4]]
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

out：

```python
Ones Tensor:
 tensor([[1, 1],
        [1, 1]])

Random Tensor:
 tensor([[0.4223, 0.1719],
        [0.3184, 0.2631]])
```



**使用随机值或常量值：**

`shape`是张量维度的元组。在下面的函数中，它确定输出张量的维数。

```python
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```

out：

```python
Random Tensor:
 tensor([[0.1602, 0.6000, 0.4126],
        [0.5558, 0.0912, 0.3004]])

Ones Tensor:
 tensor([[1., 1., 1.],
        [1., 1., 1.]])

Zeros Tensor:
 tensor([[0., 0., 0.],
        [0., 0., 0.]])
```

## 2、Tensor 的属性

Tensor 属性描述其形状、数据类型和存储它们的设备。

```python
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

out：

```
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```

## 3、对 Tensor 的操作

超过 100 种张量运算，包括算术、线性代数、矩阵操作（转置、 indexing， slicing）、sampling 等是 [此处](https://pytorch.org/docs/stable/torch.html)进行了全面描述。

这些操作中的每一个都可以在 GPU 上运行（通常比在 GPU 上的 CPU）。如果您使用的是 Colab，请转至 Runtime > Change runtime type > GPU 来分配 GPU。

默认情况下，张量是在 CPU 上创建的。我们需要使用 method 将张量显式移动到 GPU（在检查 GPU 可用性之后）。请记住，复制大型张量 跨设备在时间和内存方面可能很昂贵！`.to`

```python
# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
```

尝试列表中的一些操作。 如果您熟悉 NumPy API，您会发现 Tensor API 使用起来轻而易举。



**标准的类似 numpy 的索引和切片：**

```
tensor = torch.ones(4, 4)
print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(tensor)
```

out：

```
First row:  tensor([1., 1., 1., 1.])
First column:  tensor([1., 1., 1., 1.])
Last column: tensor([1., 1., 1., 1.])
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

out：

```
tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
```



**算术运算**

```python
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value - 计算两个张量之间的矩阵乘法
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value - 计算元素与元素之间的乘积
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
```



**单元素张量**如果您有一个单元素张量，例如，通过聚合所有 值转换为一个值，则可以将其转换为 Python 数值使用 ：`item()`

```python
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))
```

out：

```
12.0 <class 'float'>
```



**就地操作**将结果存储到操作数中的操作称为就地调用。它们由后缀表示。 例如： ， ， 将更改 .`_``x.copy_(y)``x.t_()``x`

```python
print(tensor, "\n")
tensor.add_(5)
print(tensor)
```

out：

```python
tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])
```

注意：就地操作可以节省一些内存，但在计算导数时可能会产生问题，因为会立即丢失 历史。因此，不鼓励使用它们。

## 4、使用 NumPy 桥接

CPU 和 NumPy 数组上的张量可以共享其底层内存 locations 的 Locations，更改一个位置将更改另一个位置。

#### Tensor 到 NumPy 数组

```python
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
```

out：

```python
t: tensor([1., 1., 1., 1., 1.])
n: [1. 1. 1. 1. 1.]
```

张量的变化反映在 NumPy 数组中。

```python
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
```

out：

```python
t: tensor([2., 2., 2., 2., 2.])
n: [2. 2. 2. 2. 2.]
```



#### NumPy 数组到 Tensor

```python
n = np.ones(5)
t = torch.from_numpy(n)
```

NumPy 数组中更改反映在张量中。

```python
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
```

out：

```python
t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
n: [2. 2. 2. 2. 2.]
```

## 三、Datasets & Dataloaders

用于处理数据样本的代码可能会变得混乱且难以维护;理想情况下，我们希望我们的数据集代码 与我们的模型训练代码解耦，以提高可读性和模块化。 PyTorch 提供了两个数据基元：它们允许您使用预加载的数据集以及您自己的数据。 存储样本及其相应的标签，并将 iterable 包装在 以便轻松访问样本。`torch.utils.data.DataLoader``torch.utils.data.Dataset``Dataset``DataLoader``Dataset`

PyTorch 域库提供了许多预加载的数据集（例如 FashionMNIST），这些数据集 子类并实现特定于特定数据的函数。 它们可用于对模型进行原型设计和基准测试。你可以找到它们 此处：[图像数据集](https://pytorch.org/vision/stable/datasets.html)、[文本数据集](https://pytorch.org/text/stable/datasets.html)和[音频数据集](https://pytorch.org/audio/stable/datasets.html)`torch.utils.data.Dataset`

## 1、加载数据集

以下是如何从 TorchVision 加载 [Fashion-MNIST](https://research.zalando.com/project/fashion_mnist/fashion_mnist/) 数据集的示例。 Fashion-MNIST 是 Zalando 的文章图像数据集，由 60,000 个训练示例和 10,000 个测试示例组成。 每个示例都包含一个 28×28 灰度图像和一个来自 10 个类之一的关联标签。

- 我们使用以下参数加载 [FashionMNIST 数据集](https://pytorch.org/vision/stable/datasets.html#fashion-mnist)：

  `root`是存储训练/测试数据的路径，`train`指定训练或测试数据集，`download=True`从 Internet 下载数据（如果 上没有数据）。`root``transform`并指定特征和标签转换`target_transform`

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


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
```

out：

```
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0/26421880 [00:00<?, ?it/s]
  0%|          | 65536/26421880 [00:00<01:09, 377253.59it/s]
  1%|          | 196608/26421880 [00:00<00:43, 599407.97it/s]
  3%|3         | 851968/26421880 [00:00<00:12, 2046161.58it/s]
 13%|#2        | 3375104/26421880 [00:00<00:03, 6961692.19it/s]
 36%|###5      | 9404416/26421880 [00:00<00:01, 16934518.65it/s]
 59%|#####8    | 15499264/26421880 [00:01<00:00, 23069363.05it/s]
 81%|########  | 21299200/26421880 [00:01<00:00, 26406360.70it/s]
100%|##########| 26421880/26421880 [00:01<00:00, 20091991.40it/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0/29515 [00:00<?, ?it/s]
100%|##########| 29515/29515 [00:00<00:00, 343072.27it/s]
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0/4422102 [00:00<?, ?it/s]
  1%|1         | 65536/4422102 [00:00<00:11, 375717.98it/s]
  5%|5         | 229376/4422102 [00:00<00:05, 705945.39it/s]
 21%|##1       | 950272/4422102 [00:00<00:01, 2266049.96it/s]
 87%|########6 | 3833856/4422102 [00:00<00:00, 7878221.87it/s]
100%|##########| 4422102/4422102 [00:00<00:00, 6309770.61it/s]
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0/5148 [00:00<?, ?it/s]
100%|##########| 5148/5148 [00:00<00:00, 30326231.73it/s]
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```



## 2、迭代和可视化数据集

我们可以像列表一样手动索引： . 我们用于可视化训练数据中的一些样本。`Datasets``training_data[index]``matplotlib`

```python
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
#使用 matplotlib.pyplot 创建一个新的图形，设置其大小为8x8英寸。
figure = plt.figure(figsize=(8, 8))
#设置图形中要显示的列数和行数。这里定义了一个3x3的网格，最多可以显示9个样本。
cols, rows = 3, 3
#这个循环将执行9次（从1到9），用于填充3x3的网格。
for i in range(1, cols * rows + 1):
    #使用PyTorch的 randint 函数从 training_data 中随机选择一个索引。len(training_data) 返回训练数据的总样本数，size=(1,) 表示生成一个随机整数，.item()则将返回的张量转为Python整数。
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    #通过随机选择的 sample_idx 从 training_data 中提取出对应的图像 img 和标签 label。
    img, label = training_data[sample_idx]
    #向图形中添加一个子图，行数和列数为之前定义的值，i表示当前子图的位置。
    figure.add_subplot(rows, cols, i)
    #设置当前子图的标题，使用 labels_map 映射对 label 进行解码，以获取相应的类别名称。
    plt.title(labels_map[label])
    #隐藏当前子图的坐标轴，使得图像更加清晰可见。
    plt.axis("off")
    #使用 imshow 显示图像，.squeeze() 方法用于去掉单维度（如灰度图像的通道维度），cmap="gray" 指定使用灰度色彩图。
    plt.imshow(img.squeeze(), cmap="gray")
#显示整个图形，包括所有子图，将结果呈现出来。
plt.show()
```

<img src="C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241008154307017.png" alt="image-20241008154307017" style="zoom:67%;" />



## 3、为您的文件创建自定义数据集

自定义 Dataset 类必须实现三个函数：__init__、__len__ 和 __getitem__。 看看这个实现;存储 FashionMNIST 图像 在 directory 中，并且它们的标签单独存储在 CSV file 中。`img_dir``annotations_file`

在接下来的部分中，我们将分解每个函数中发生的情况。

```python
#os：用于处理文件和目录名，主要用来构建图像路径。
#pandas：用于数据处理，尤其是数据框的操作，这里用于读取 CSV 文件。
#read_image：来自 torchvision.io，用于读取图像文件并将其转换为 PyTorch 张量。
import os
import pandas as pd
from torchvision.io import read_image

#定义自定义数据集类：继承自 Dataset（通常是PyTorch中的一个抽象基类），表示这是一个数据集的自定义实现。
class CustomImageDataset(Dataset):
    #初始化方法：这是构造函数，用于初始化对象。annotations_file: CSV 文件的路径，包含图像标签信息。
#img_dir: 存放图像的目录。transform: 图像预处理的转换(默认为None)。target_transform: 标签预处理的转换(默认为None)。
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        #读取标签信息：使用 pandas 读取 CSV 文件，结果存储在 img_labels 中，通常会包含图像文件名和相应的标签。
        self.img_labels = pd.read_csv(annotations_file)
        #存储图像目录：将传入的目录路径存储为类的属性，后续用于构建完整的图像路径。
        self.img_dir = img_dir
        #存储转换函数：将可选的图像和标签转换函数存储为类的属性，以便在获取每个图像和标签时应用这些转换。
        self.transform = transform
        self.target_transform = target_transform

    #获取数据集大小：重写 __len__ 方法，使得可以使用 len(dataset) 获取数据集的样本数量。
    def __len__(self):
        return len(self.img_labels)

    #获取指定索引的样本：重写 __getitem__ 方法，以便通过索引访问数据集中每个样本。
    def __getitem__(self, idx):
        #构建图像路径：使用 os.path.join 创建完整的图像路径。self.img_labels.iloc[idx, 0] 获取当前索引 idx 的图像文件名。
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        #读取图像：调用 read_image 读取指定路径的图像，并返回一个 PyTorch 张量。
        image = read_image(img_path)
        #获取标签：从标签信息中获取当前索引 idx 的标签。
        label = self.img_labels.iloc[idx, 1]
        #应用图像转换（如果有）：如果传入了图像转换函数，则对图像进行转换（例如，归一化、裁剪等）。
        if self.transform:
            image = self.transform(image)
        #应用标签转换（如果有）：如果传入了标签转换函数，则对标签进行转换（例如，类别编码等）。
        if self.target_transform:
            label = self.target_transform(label)
        #返回图像和标签：返回处理后的图像和标签，供后续的模型训练或测试使用。
        return image, label
```



### `__init__`

__init__ 函数在实例化 Dataset 对象时运行一次。我们初始化 包含图像、annotations 文件和两个转换的目录（覆盖 在下一节中将有更详细的介绍）。

labels.csv 文件如下所示：

```
tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9
```



```python
#初始化方法：这是构造函数，用于初始化对象。annotations_file: CSV 文件的路径，包含图像标签信息。
#img_dir: 存放图像的目录。transform: 图像预处理的转换(默认为None)。target_transform: 标签预处理的转换(默认为None)。
def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    #读取标签信息：使用pandas读取CSV文件，结果存储在img_labels中，通常会包含图像文件名和相应的标签。
    self.img_labels = pd.read_csv(annotations_file)
    #存储图像目录：将传入的目录路径存储为类的属性，后续用于构建完整的图像路径。
    self.img_dir = img_dir
    #存储转换函数：将可选的图像和标签转换函数存储为类的属性，以便在获取每个图像和标签时应用这些转换。
    self.transform = transform
    self.target_transform = target_transform
```



### `__len__`

__len__ 函数返回数据集中的样本数。

例：

```python
#获取数据集大小：重写 __len__ 方法，使得可以使用 len(dataset) 获取数据集的样本数量。
def __len__(self):
    return len(self.img_labels)
```



### `__getitem__`

__getitem__ 函数加载并返回位于给定 index 处的数据集中的样本。 根据索引，它识别图像在磁盘上的位置，使用 将其转换为张量，检索 中 CSV 数据的相应标签，对它们调用 transform 函数（如果适用），并返回 Tensor 图像和元组中的相应标签。`idx``read_image``self.img_labels`

```python
#获取指定索引的样本：重写 __getitem__ 方法，以便通过索引访问数据集中每个样本。
def __getitem__(self, idx):
    #构建图像路径：使用 os.path.join 创建完整的图像路径。self.img_labels.iloc[idx, 0] 获取当前索引 idx 的图像文件名。
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    #读取图像：调用 read_image 读取指定路径的图像，并返回一个 PyTorch 张量。
    image = read_image(img_path)
    #获取标签：从标签信息中获取当前索引 idx 的标签。
    label = self.img_labels.iloc[idx, 1]
    #应用图像转换（如果有）：如果传入了图像转换函数，则对图像进行转换（例如，归一化、裁剪等）。
    if self.transform:
        image = self.transform(image)
    #应用标签转换（如果有）：如果传入了标签转换函数，则对标签进行转换（例如，类别编码等）。
    if self.target_transform:
        label = self.target_transform(label)
    #返回图像和标签：返回处理后的图像和标签，供后续的模型训练或测试使用。
    return image, label
```



## 4、准备数据以使用 DataLoader 进行训练

它会检索我们数据集的特征，并一次标记一个样本。在训练模型时，我们通常希望 以 “小批量” 传递样本，在每个 epoch 重新洗牌数据以减少模型过拟合，并使用 Python 的 加快数据检索速度。`Dataset``multiprocessing`

`DataLoader`是一个可迭代对象，它通过一个简单的 API 为我们抽象了这种复杂性。

```python
#导入 DataLoader：从 torch.utils.data 模块导入 DataLoader 类，它用于处理数据集的批量加载。
from torch.utils.data import DataLoader

#创建数据加载器：
#train_dataloader 和 test_dataloader：分别为训练和测试数据集的加载器。
#training_data 和 test_data 是包含图像和标签的自定义数据集对象。
#batch_size=64：每个批次加载64个样本。
#shuffle=True：在每个epoch开始时随机打乱数据，以提高训练的多样性和模型的泛化能力。
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```



## 5、遍历 DataLoader

我们已将该数据集加载到 中，并可以根据需要迭代数据集。 下面的每次迭代都会返回一批 and（分别包含特征和标签）。 因为我们指定了 ，所以在我们迭代所有 batchs 之后，数据会被随机排序（以便对 数据加载顺序，看看 [Samplers](https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler)）。`DataLoader``train_features``train_labels``batch_size=64``shuffle=True`

```python
#提取一个批次样本：iter(train_dataloader) 创建一个迭代器，next(...) 获取该迭代器下一批数据。
#train_features：存储提取的训练样本（图像）。
#train_labels：存储提取的相应标签。
#train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
#打印批次信息：train_features.size() 和 train_labels.size() 输出当前批次特征和标签的形状，便于了解模型输入和输出的维度。
img = train_features[0].squeeze()
#提取第一张图像：获取当前批次中的第一张图像，并使用 squeeze() 方法去除可能的单维度（例如，对于灰度图像，通道数可能是1）。
label = train_labels[0]
#提取第一张图像的标签：获取当前批次中第一张图像对应的标签
plt.imshow(img, cmap="gray")
#可视化图像：使用 matplotlib.pyplot 中的 imshow 显示提取的图像，cmap="gray" 指定以灰度图形式显示（适用于较好的图像可视化）。
plt.show()
#展示当前图像：显示已设置的图像。
print(f"Label: {label}")
```

![数据教程](https://pytorch.org/tutorials/_images/sphx_glr_data_tutorial_002.png)

out：

```python
Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
Label: 5
```



## 四、变换

数据并不总是以所需的最终处理形式出现 训练机器学习算法。我们使用 **transform** 来执行一些 操作数据并使其适合训练。

所有 TorchVision 数据集都有两个参数 - 修改特征和修改标签 - 接受包含转换逻辑的可调用对象。 [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html) 模块提供 几个开箱即用的常用转换。`transform``target_transform`

FashionMNIST 功能采用 PIL 图像格式，标签为整数。 对于训练，我们需要将**特征作为标准化张量，将标签作为独热编码张量**。 为了进行这些转换，我们使用 和 。`ToTensor``Lambda`

```python
#torch：PyTorch 库，用于构建和训练深度学习模型。
#datasets：从 torchvision 中导入的子模块，提供常用的数据集，包括 FashionMNIST。
#ToTensor：从 torchvision.transforms 导入，用于将 PIL 图像或 NumPy 数组转换为 PyTorch 张量。
#Lambda：允许使用任意函数进行转换，特别是针对目标标签的自定义转换。
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

'''
加载 FashionMNIST 数据集：
root="data"：指定数据集存放的根目录。如果这个目录不存在，数据集将被下载到这里。
train=True：指明加载训练集。如果设置为 False，则加载测试集。
download=True：如果数据集在指定的 root 目录中不存在，它会自动下载。
transform=ToTensor()：将加载的图像转换为 PyTorch 张量。所有图像的像素值将被缩放到 [0, 1] 之间，原始的 [0, 255] 范围将被规范化。
'''
ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    '''
    目标转换：
	target_transform：指定如何转换目标（即标签）。
	Lambda(lambda y: ...)：定义一个匿名函数，对标签进行转换。
	torch.zeros(10, dtype=torch.float)：创建一个大小为10的零张量，表示10个类（在FashionMNIST
	中，标签为0到9）。
	scatter_(0, torch.tensor(y), value=1)：在第0维（即行）上将位置为 y 的值置为1，实现了将整数
	标签转换为独热编码（one-hot encoding）。
	y 是当前样本的标签（如0，1，2等）。
	这个过程将标签 y（如2）转换为张量 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]，表示样本属于类2。
    '''
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```

out：

```
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0/26421880 [00:00<?, ?it/s]
  0%|          | 65536/26421880 [00:00<01:10, 371929.58it/s]
  0%|          | 131072/26421880 [00:00<01:10, 372956.07it/s]
  1%|          | 262144/26421880 [00:00<00:48, 540145.75it/s]
  1%|1         | 360448/26421880 [00:00<00:47, 547218.26it/s]
  2%|1         | 491520/26421880 [00:00<00:42, 617052.92it/s]
  2%|2         | 622592/26421880 [00:01<00:39, 660011.39it/s]
  3%|2         | 753664/26421880 [00:01<00:37, 689537.15it/s]
  3%|3         | 917504/26421880 [00:01<00:33, 765886.58it/s]
  4%|4         | 1114112/26421880 [00:01<00:28, 873157.18it/s]
  5%|4         | 1310720/26421880 [00:01<00:26, 948067.93it/s]
  6%|5         | 1507328/26421880 [00:01<00:24, 1002435.45it/s]
  7%|6         | 1769472/26421880 [00:02<00:21, 1146641.38it/s]
  8%|7         | 2031616/26421880 [00:02<00:19, 1249934.10it/s]
  9%|8         | 2326528/26421880 [00:02<00:17, 1376347.07it/s]
 10%|9         | 2621440/26421880 [00:02<00:15, 1565875.17it/s]
 11%|#         | 2818048/26421880 [00:02<00:15, 1568546.80it/s]
 12%|#2        | 3178496/26421880 [00:02<00:13, 1727699.09it/s]
 14%|#3        | 3571712/26421880 [00:03<00:11, 2026833.11it/s]
 14%|#4        | 3801088/26421880 [00:03<00:11, 1979301.38it/s]
 16%|#6        | 4292608/26421880 [00:03<00:09, 2252653.32it/s]
 18%|#8        | 4816896/26421880 [00:03<00:08, 2490218.21it/s]
 20%|##        | 5373952/26421880 [00:03<00:07, 2708732.89it/s]
 23%|##2       | 6029312/26421880 [00:03<00:06, 3023060.61it/s]
 25%|##5       | 6717440/26421880 [00:04<00:05, 3297915.07it/s]
 28%|##8       | 7503872/26421880 [00:04<00:05, 3653304.51it/s]
 32%|###1      | 8355840/26421880 [00:04<00:04, 4014052.40it/s]
 35%|###5      | 9273344/26421880 [00:04<00:03, 4375261.31it/s]
 39%|###9      | 10321920/26421880 [00:04<00:03, 4849845.41it/s]
 43%|####3     | 11468800/26421880 [00:04<00:02, 5344851.29it/s]
 48%|####8     | 12713984/26421880 [00:05<00:02, 5865372.51it/s]
 53%|#####3    | 14123008/26421880 [00:05<00:01, 6500251.82it/s]
 59%|#####9    | 15630336/26421880 [00:05<00:01, 7120882.70it/s]
 66%|######5   | 17334272/26421880 [00:05<00:01, 7866648.91it/s]
 73%|#######2  | 19202048/26421880 [00:05<00:00, 8679428.81it/s]
 80%|########  | 21233664/26421880 [00:05<00:00, 9546374.43it/s]
 89%|########8 | 23461888/26421880 [00:06<00:00, 11204286.11it/s]
 94%|#########3| 24707072/26421880 [00:06<00:00, 10857261.29it/s]
100%|##########| 26421880/26421880 [00:06<00:00, 4161767.03it/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0/29515 [00:00<?, ?it/s]
100%|##########| 29515/29515 [00:00<00:00, 337880.12it/s]
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0/4422102 [00:00<?, ?it/s]
  1%|1         | 65536/4422102 [00:00<00:11, 376496.60it/s]
  5%|5         | 229376/4422102 [00:00<00:05, 707348.79it/s]
 21%|##1       | 950272/4422102 [00:00<00:01, 2270129.95it/s]
 87%|########6 | 3833856/4422102 [00:00<00:00, 7881599.71it/s]
100%|##########| 4422102/4422102 [00:00<00:00, 6318633.21it/s]
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0/5148 [00:00<?, ?it/s]
100%|##########| 5148/5148 [00:00<00:00, 32715571.20it/s]
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```

### ToTensor（）

[ToTensor](https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.ToTensor) 将 PIL 图像或 NumPy 转换为 .和秤 图像的像素强度值在 [0.， 1.] 范围内。`ndarray``FloatTensor`

将加载的图像转换为PyTorch张量。所有图像的像素值将被缩放到[0,1]之间，原始的[0, 255]范围将被规范化。

### Lambda 转换

Lambda 转换应用任何用户定义的 lambda 函数。在这里，我们定义了一个函数 将整数转换为 one-hot 编码张量。 它首先创建一个大小为 10 的零张量（我们数据集中的标签数量）并调用 [scatter_](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html) 在标签 .`value=1``y`

```python
target_transform = Lambda(lambda y: torch.zeros(
    10, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
```





## 五、构建神经网络

神经网络由对数据执行操作的层/模块组成。 [torch.nn](https://pytorch.org/docs/stable/nn.html) 命名空间提供了您需要的所有构建块 构建您自己的神经网络。PyTorch 中的每个模块都对 [nn.模块](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)。 神经网络本身是一个由其他模块（层）组成的模块。这种嵌套结构允许 轻松构建和管理复杂的架构。

在以下部分中，我们将构建一个神经网络来对 FashionMNIST 数据集中的图像进行分类。

```python
'''
导入os模块：os模块提供了与操作系统交互的功能。常用于文件和目录的操作，如创建文件夹、删除文件、路径操作等。
导入torch：这是PyTorch的主要模块，提供了构建和训练深度学习模型所需的基本功能，如张量操作、自动微分等。
从torch导入nn模块：nn模块包含构建神经网络的工具和组件，如各类层（例如全连接层、卷积层等）、损失函数等。使用nn模块，可以简化网络模型的构建过程。
从torch.utils.data导入DataLoader：DataLoader是用于批量加载数据的工具。它能够将数据集分成小批次，支持多线程加载，方便用于训练和测试模型。
从torchvision导入datasets和transforms：
datasets：包含许多常见的数据集的实现，例如MNIST、CIFAR-10等，因此可以便捷地进行加载和处理。
transforms：提供数据预处理和数据增强的功能，如归一化、裁剪、旋转等。在使用深度学习训练时，常常对输入数据进行一些变换，以提高模型的泛化能力。
'''
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```



## 1、获取用于训练的设备

我们希望能够在 GPU 或 MPS 等硬件加速器上训练我们的模型。 如果可用。让我们检查一下 [torch.cuda](https://pytorch.org/docs/stable/notes/cuda.html) 或 [torch.backends.mps](https://pytorch.org/docs/stable/notes/mps.html) 是否可用，否则我们使用 CPU。

```python
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
```

out：

```
Using cuda device
```



## 2、定义类

我们通过子类化 来定义我们的神经网络，并且 在 中初始化神经网络层。每个子类都实现了 对方法中输入数据的操作。`nn.Module``__init__``nn.Module``forward`

```python
#定义神经网络类：NeuralNetwork 继承自 nn.Module，这是所有神经网络模块的基类。通过继承它，可以使用 PyTorch 提供的丰富功能。
class NeuralNetwork(nn.Module):
    #初始化方法：__init__ 是类的构造函数，用于初始化网络的结构。
	#super().__init__()：调用父类 nn.Module 的构造函数，以便初始化PyTorch相关的属性。
    def __init__(self):
        super().__init__()
        #扁平化层：创建一个 Flatten 层，它将输入的多维张量（如 28x28 的图像）转换为一维张量（例如，784 维）。这个步骤是必要的，因为全连接层（如 nn.Linear）期望输入是一维张量。
        self.flatten = nn.Flatten()
    '''
    创建线性层和激活函数的顺序：
	nn.Sequential：这是一个有序容器，允许将多个子模块（如层）组合成一个模块。
	nn.Linear(28*28, 512)：输入层，分别有 784 个输入（图像的像素数量）和 512 个输出（隐藏层的单元
	数）。
	nn.ReLU()：后接一个 ReLU（Rectified Linear Unit）激活函数。它将负值置为零，帮助网络学习非线性
	关系。
	另一个 nn.Linear(512, 512)：第二个隐藏层，继续有 512 个输入和 512 个输出。
	再次使用 ReLU 激活函数。
	nn.Linear(512, 10)：输出层，有 10 个输出，对应于 10 类（例如，分类问题中的 10 种不同标签）。
    '''
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

        
    #定义前向传播方法：forward 方法定义了输入如何通过网络进行传播。这是神经网络的核心机制之一。
    def forward(self, x):
        #扁平化输入：将输入 x（可能是一个批次的图像）扁平化为一维张量。
        x = self.flatten(x)
        #通过线性-ReLU 堆栈进行前向传播：将扁平化后的输入传递给 linear_relu_stack，计算出网络的输出（ logits）。这些输出值表示每个类的原始得分，通常在进行分类任务时使用 softmax 函数进行概率转换。
        logits = self.linear_relu_stack(x)
        #返回输出：返回网络的输出（logits），以便后续进行损失计算和优化。
        return logits
```



我们创建一个实例 ，并将其移动到 中，然后打印 它的结构。`NeuralNetwork``device`

```python
model = NeuralNetwork().to(device)
print(model)
```

out：

```
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
```



要使用该模型，我们将输入数据传递给它。这将执行模型的 ， 以及一些[后台操作](https://github.com/pytorch/pytorch/blob/270111b7b611d174967ed204776985cefca9c144/torch/nn/modules/module.py#L866)。 不要直接打电话！`forward``model.forward()`

在输入上调用模型将返回一个二维张量，其中 dim=0 对应于每个类的 10 个原始预测值的每个输出，dim=1 对应于每个输出的单个值。 我们通过模块的一个实例传递预测概率来获得预测概率。`nn.Softmax`

```python
'''
torch.rand(1, 28, 28, device=device)：
这里调用了 torch.rand 函数，生成一个随机的张量。
这条语句创建一个形状为 (1, 28, 28) 的张量，表示一个包含1个样本的28x28的图像。常用于模拟图像数据，例如手写数字图片（如MNIST数据集）。
device=device 表示这个张量会被创建在之前定义的 device（可能为 CPU 或 GPU）上。
'''
X = torch.rand(1, 28, 28, device=device)
'''
model(X)：
这里将随机输入 X 传入神经网络模型 model，进行前向传播计算。
模型将输出一个张量 logits，该张量的形状通常为 (1, num_classes)，其中 num_classes 是模型最后一层的输出类别数（例如，对于 MNIST 数据集，num_classes 应该是10，因为数字从 0 到 9）。
'''
logits = model(X)
'''
nn.Softmax(dim=1)：
这里创建了一个 Softmax 激活函数的实例。Softmax 用于将 logits 转换为概率分布。
dim=1 表示沿着第1维进行归一化，即对每个样本（在这里是只有1个样本的情况）计算概率，确保输出的所有概率和为1。
输出：pred_probab 现在是一个形状为 (1, num_classes) 的张量，其中每个元素表示特定类别的预测概率。
'''
pred_probab = nn.Softmax(dim=1)(logits)
'''
pred_probab.argmax(1)：
argmax 函数返回指定维度上最大值的索引。这里 argmax(1) 表示沿着维度1查找最大值的索引（即类别的索引）。
结果 y_pred 将是一个张量，包含对应于每个样本的预测类别的标签。
'''
y_pred = pred_probab.argmax(1)
'''
print(f"Predicted class: {y_pred}")：
通过格式化字符串将预测类别输出到控制台。
'''
print(f"Predicted class: {y_pred}")
```

out：

```
Predicted class: tensor([7], device='cuda:0')
```



## 3、模型层

让我们分解 FashionMNIST 模型中的层。为了说明这一点，我们 将获取 3 张大小为 28x28 的图像的示例小批量，并查看它会发生什么变化 我们通过网络传递它。

```python
# 创建包含3张28x28的随机图像
input_image = torch.rand(3,28,28)
#打印张量的尺寸
print(input_image.size())
```

out：

```
torch.Size([3, 28, 28])
```



## 4、nn.Flatten() - 扁平化

我们初始化 [nn.拼合](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html)图层，将每个 2D 28x28 图像转换为包含 784 个像素值的连续数组 （ 保持 Minibatch 维度（dim=0 时）。

```python
'''
nn.Flatten 是 PyTorch 中的一个层，用于将输入的多维张量转换为一维张量。它的作用是将输入的形状（如图像的形状）展平，通常用于将图像数据输入到全连接层（线性层）之前。
'''
flatten = nn.Flatten()
'''
当你调用 flatten(input_image) 时，nn.Flatten() 会将每张 28x28 的图像展平。每张图像的 28x28 像素会被展平为一个长度为 784（即 28×28=784 ）的一维数组。
由于你有 3 张图像，因此最终的输出张量的形状会是 (3, 784)。
'''
flat_image = flatten(input_image)
#打印张量的尺寸
print(flat_image.size())
```

out：

```
torch.Size([3, 784])
```

**注：此处的（3，784），相当于是将二维张量转换成一维张量，其中一维张量中存储了一个784维的向量**

**扁平化层解释说明：**

![image-20241008174410495](C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241008174410495.png)



## 5、nn.Linear - 线性变换

[线性层](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)是一个模块，它使用其存储的权重和偏差对输入应用线性变换。

```python
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
```

out：

```
torch.Size([3, 20])
```

**nn.Linear的输入输出转换原理：**

![image-20241008195020764](C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241008195020764.png)



## 6、nn.ReLU() - 激活函数

非线性激活是在模型的输入和输出之间创建复杂的映射。 它们在线性变换后应用以引入*非线性*，从而帮助神经网络 学习各种各样的现象。

在这个模型中，我们使用 [nn.ReLU](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html) 之间的 线性层，但还有其他激活会在模型中引入非线性。

**nn.ReLu()激活函数的作用和意义：**

![image-20241008195240677](C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241008195240677.png)



## 7、nn.Sequential - 有序模块

[nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html) 是有序的 模块的容器。数据按照定义的顺序通过所有模块传递。您可以使用 顺序容器将 .`seq_modules`

```python
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
```



## 8、nn.Softmax - 转换概率分布函数

神经网络的最后一个线性层返回 logits（[-infty， infty] 中的原始值），这些值将传递给 [nn.Softmax](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html) 模块。logit 将缩放为值 [0， 1] 表示模型对每个类的预测概率。 parameter 指示沿 的值之和必须为 1。`dim`

```python
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
```

**nn.Softmax(dim=1)计算概率的原理与解释：**

![image-20241008195701168](C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241008195701168.png)



## 9、模型参数

神经网络中的许多层都是*参数化*的，即具有相关的权重 以及在训练期间优化的偏差。自动子类化 跟踪模型对象中定义的所有字段，并生成所有参数 可使用您的模型或方法访问。`nn.Module``parameters()``named_parameters()`

在此示例中，我们遍历每个参数，并打印其大小和值的预览。

```python
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
```



## 六、自动微分`torch.autograd`

在训练神经网络时，最常用的算法是**反向传播**。在此算法中，参数 （模型权重） 为 根据 loss 函数的**梯度**进行调整 添加到给定的参数中。

为了计算这些梯度，PyTorch 有一个内置的微分引擎 叫。它支持自动计算任何 计算图。`torch.autograd`

考虑最简单的单层神经网络，输入 ， parameters 和 ，以及一些 loss 函数。它可以在 PyTorch 的调用方法如下：`x``w``b`

```python
import torch

# x 是一个包含 5 个元素的张量，其中所有元素的值均为 1。它代表模型的输入。
x = torch.ones(5)  # input tensor

# y 是一个包含 3 个元素的张量，其中所有元素的值均为 0。这通常代表模型期望的输出或者真实标签。
y = torch.zeros(3)  # expected output

#w是一个形状为(5,3)的张量，代表权重。使用 torch.randn 随机生成元素，元素值符合标准正态分布（均值为0，方差为1）。
#requires_grad=True 表示在后续的计算中，PyTorch将记录对于这个张量的所有操作，以便之后进行自动求导。
w = torch.randn(5, 3, requires_grad=True)

#b是一个形状为(3,)的张量，代表偏置。生成方式与权重相同，也是随机得来，并同样设置requires_grad=True。
b = torch.randn(3, requires_grad=True)

'''
torch.matmul(x, w)：这里使用点积（矩阵乘法）计算输入张量 x 和权重张量 w 的乘积。结果是一个形状为 (3,) 的张量，因为 x 的尺寸是 (5,)，而 w 的尺寸是 (5, 3)。
+ b：将偏置 b 添加到结果中，得到 z，它也是形状为 (3,) 的张量。
类似nn.Linear操作
'''
z = torch.matmul(x, w)+b

'''
这个函数计算二进制交叉熵损失，它是在计算 sigmoid 函数后应用于 logits（未经激活的模型输出）。这里，z 代表模型输出，y 是实际标签。
函数内部会先对 z 应用 sigmoid 函数，然后计算损失。
'''
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
```



## 1、张量、函数和计算图

此代码定义以下**计算图**：

![image-20241008204747097](C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241008204747097.png)

## 2、计算梯度

为了优化神经网络中参数的权重，我们需要 计算我们的损失函数关于参数的导数，和一些固定值下。为了计算这些导数，我们调用 ，然后从 和 中检索值：`x``y``loss.backward()``w.grad``b.grad`

```python
loss.backward()
print(w.grad)
print(b.grad)
```

![image-20241008204910089](C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241008204910089.png)

## 3、禁用渐变跟踪

默认情况下，所有具有 的张量都在跟踪其 计算历史并支持梯度计算。然而，那里 在某些情况下，我们不需要这样做，例如，当我们有 训练了模型，只想将其应用于一些输入数据，即我们 只想通过网络进行*前向*计算。我们可以停止 通过使用 Block 包围我们的计算代码来跟踪计算：`requires_grad=True``torch.no_grad()`

```python
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
```

实现相同结果的另一种方法是使用 在 Tensor 上：`detach()`

```python
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)
```

出于多种原因，您可能希望禁用渐变跟踪：

- 将神经网络中的某些参数标记为**冻结参数**。
- 在仅执行前向传递时**加快计算**速度，因为对执行 不跟踪渐变会更有效。

```python
'''
1. 默认梯度跟踪
在 PyTorch 中，当张量的 requires_grad 属性设置为 True 时，这些张量会被自动跟踪其计算历史。这意味着 PyTorch 会记录其所有操作，以便在调用 .backward() 时计算梯度。这对于模型训练时是非常重要的。

2. 禁用梯度跟踪的需求
然而，在某些特定情况下，我们可能不需要梯度计算，比如：
--推理阶段（Inference）：当我们已经训练了模型，只希望将其应用于新输入数据时，我们只想进行前向计算，而不需要计算梯度。
--提高性能：在不需要梯度的情况下禁用跟踪可以提高计算速度，因为会减少额外的内存和计算开销。
'''
```



## 七、优化模型参数

现在我们已经有了模型和数据，是时候通过优化模型的参数来训练、验证和测试我们的模型了 我们的数据。训练模型是一个迭代过程;在每次迭代中，模型都会对输出进行猜测，计算 其 guess （*loss*） 中的误差收集了误差相对于其参数的导数（正如我们在 [上一节](https://pytorch.org/tutorials/beginner/basics/autograd_tutorial.html)），并使用梯度下降**优化**这些参数。如需更多 此过程的详细演练，请观看 [3Blue1Brown 的反向传播](https://www.youtube.com/watch?v=tIeHLnjs5U8)视频。

## 1、先决条件代码

我们从前面的数据集[和数据加载器](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)部分加载代码，并[构建模型](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)。

```python
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
```

out：

```
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz

  0%|          | 0/26421880 [00:00<?, ?it/s]
  0%|          | 65536/26421880 [00:00<01:09, 377467.03it/s]
  1%|          | 229376/26421880 [00:00<00:36, 709640.53it/s]
  4%|3         | 950272/26421880 [00:00<00:11, 2277818.25it/s]
 15%|#4        | 3833856/26421880 [00:00<00:02, 7918883.56it/s]
 38%|###7      | 9994240/26421880 [00:00<00:00, 17817569.33it/s]
 61%|######1   | 16154624/26421880 [00:01<00:00, 23783205.05it/s]
 85%|########4 | 22380544/26421880 [00:01<00:00, 27650599.23it/s]
100%|##########| 26421880/26421880 [00:01<00:00, 20153562.49it/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz

  0%|          | 0/29515 [00:00<?, ?it/s]
100%|##########| 29515/29515 [00:00<00:00, 342514.13it/s]
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz

  0%|          | 0/4422102 [00:00<?, ?it/s]
  1%|1         | 65536/4422102 [00:00<00:11, 375405.49it/s]
  5%|5         | 229376/4422102 [00:00<00:05, 704999.19it/s]
 21%|##        | 917504/4422102 [00:00<00:01, 2178667.67it/s]
 83%|########2 | 3670016/4422102 [00:00<00:00, 7522364.27it/s]
100%|##########| 4422102/4422102 [00:00<00:00, 6302547.20it/s]
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz

  0%|          | 0/5148 [00:00<?, ?it/s]
100%|##########| 5148/5148 [00:00<00:00, 39985698.13it/s]
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```

## 2、超参数

超参数是可调整的参数，可让您控制模型优化过程。 不同的超参数值会影响模型训练和收敛速率 （[阅读有关](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html)超参数优化的更多信息）

- 我们定义以下用于训练的超参数：

  **Number of Epochs** - 迭代数据集的次数**Batch Size** - 在更新参数之前通过网络传播的数据样本数**学习率** - 在每个批次/epoch 更新模型参数的量。较小的值会导致学习速度变慢，而较大的值可能会导致训练期间出现不可预知的行为。

```python
learning_rate = 1e-3
batch_size = 64
epochs = 5
```

## 3、优化循环

设置超参数后，我们就可以使用优化循环来训练和优化我们的模型。每 优化循环的迭代称为 **epoch**。

- 每个 epoch 由两个主要部分组成：

  **训练循环** - 迭代训练数据集并尝试收敛到最佳参数。**验证/测试循环** - 迭代测试数据集以检查模型性能是否正在提高。

让我们简要熟悉一下训练循环中使用的一些概念。跳转到 请参阅 [优化循环的完整实现](https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html#full-impl-label) 。

### 损失函数

当看到一些训练数据时，我们未经训练的网络可能无法给出正确的 答。**损失函数**测量获得的结果与目标值的差异程度， 这是我们在训练过程中想要最小化的损失函数。为了计算损失，我们做了一个 prediction 的 Alpha 数据 Sample，并将其与 True Data Label 值进行比较。

常见的损失函数包括 [nn.MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss)（均方误差）用于回归任务，nn[.用于分类的 NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss) （Negative Log Likelihood）。[nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss) 结合了 和 。`nn.LogSoftmax``nn.NLLLoss`

我们将模型的输出 logits 传递给 ，后者将对 logit 进行归一化并计算预测误差。`nn.CrossEntropyLoss`

```python
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
```

![image-20241008211518047](C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241008211518047.png)

### 优化

优化是调整模型参数以减少每个训练步骤中的模型误差的过程。**优化算法**定义了此过程的执行方式（在此示例中，我们使用随机梯度下降）。 所有优化逻辑都封装在对象中。在这里，我们使用 SGD 优化器;此外，PyTorch 中还提供了[许多不同的优化器](https://pytorch.org/docs/stable/optim.html)，例如 ADAM 和 RMSProp，它们更适合不同类型的模型和数据。`optimizer`

我们通过注册需要训练的模型参数并传入学习率超参数来初始化优化器。

```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

![image-20241008211750645](C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241008211750645.png)

- 在训练循环中，优化分三个步骤进行：

  调用 重置模型参数的梯度。默认情况下，梯度累加;为了防止重复计数，我们在每次迭代时都将它们显式归零。`optimizer.zero_grad()`通过调用 .PyTorch 会根据每个参数来存储损失的梯度。`loss.backward()`一旦我们有了梯度，我们就会调用以通过在 backward pass 中收集的梯度来调整参数。`optimizer.step()`

## 4、全面实施

我们定义了 that 在优化代码上循环，并且 根据我们的测试数据评估模型的性能。`train_loop``test_loop`

```python
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
```

#### 总结

- `train_loop` 函数负责模型的训练过程，其中包含前向传播、损失计算、反向传播和参数更新等步骤。
- `test_loop` 函数则用于评估模型在测试集的表现，计算并输出准确率和平均损失。
- 两个函数都关注模型的训练和评估，包括适当的模式设置和不计算梯度（在测试时）以优化性能。



我们初始化损失函数和优化器，并将其传递给 和 。 随意增加 epoch 的数量来跟踪模型的改进性能。`train_loop``test_loop`

```python
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
```

![image-20241008223027864](C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241008223027864.png)



## 八、保存并加载模型

在本节中，我们将了解如何通过保存、加载和运行模型预测来保持模型状态。

```python
import torch
import torchvision.models as models
```

### 保存和加载模型权重

PyTorch 模型将学习到的参数存储在内部的 状态字典，称为 .这些可以通过以下方法持久化：`state_dict``torch.save`

```python
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')
```

要加载模型权重，您需要先创建同一模型的实例，然后加载参数 using 方法。`load_state_dict()`

在下面的代码中，我们设置了将 在解封期间执行的函数更改为仅 装载重量。使用 加载砝码时的最佳实践。`weights_only=True``weights_only=True`

```python
model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()
```

- **注意**

请务必在推理之前调用 method 以将 dropout 和 batch normalization 层设置为评估模式。如果不这样做，将产生不一致的推理结果。`model.eval()`

### 保存和加载带有形状的模型

在加载模型权重时，我们需要先实例化模型类，因为类 定义网络的结构。我们可能希望将这个类的结构与 模型，在这种情况下，我们可以将 （而不是 ） 传递给 saving 函数：`model``model.state_dict()`

```python
torch.save(model, 'model.pth')
```

然后，我们可以加载模型，如下所示。

如[保存和加载 torch.nn.Modules](https://pytorch.org/tutorials/beginner/basics/pytorch.org/docs/main/notes/serialization.html#saving-and-loading-torch-nn-modules) 中所述， saving 的 County，因为这涉及加载 模型，这是 .`state_dict``s is considered the best practice. However, below we use ``weights_only=False``torch.save`

```
model = torch.load('model.pth', weights_only=False),
```

- **注意**

这种方法在序列化模型时使用 Python [pickle](https://docs.python.org/3/library/pickle.html) 模块，因此它依赖于加载模型时可用的实际类定义。









