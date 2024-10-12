# 李沐 - 动手学深度学习 PyTorch版

### N维数组是机器学习和神经网络的主要数据结构

<img src="C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241012104908520.png" alt="image-20241012104908520" style="zoom:67%;" />

<img src="C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241012104921841.png" alt="image-20241012104921841" style="zoom: 67%;" />

### 创建数组需要

- 形状：例如3x4矩阵
- 每个元素的数据类型：例如32位浮点数
- 每个元素的值，例如全是0，或者随机数

<img src="C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241012105116779.png" alt="image-20241012105116779" style="zoom:67%;" />



<img src="C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241012105313928.png" alt="image-20241012105313928" style="zoom:67%;" />

### 将导数拓展到向量

#### 1、x为向量，y为标量，其结果y对x的偏导数为行向量（横着）- 分子布局

<img src="C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241012110206350.png" alt="image-20241012110206350" style="zoom: 50%;" />

#### 2、y为向量，x为标量，其结果y对x的偏导数为列向量（竖着）- 分母布局

<img src="C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241012110409654.png" alt="image-20241012110409654" style="zoom: 50%;" />

#### 3、y为向量，x为向量，其结果y对x的偏导数为矩阵

<img src="C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241012110533611.png" alt="image-20241012110533611" style="zoom:50%;" />

#### 梯度指向值变化最大的方向（此处T均表示为行向量）- x为向量，y为标量的样例

<img src="C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241012110015342.png" alt="image-20241012110015342" style="zoom: 50%;" />

#### 为向量，x为向量的样例

<img src="C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241012110913059.png" alt="image-20241012110913059" style="zoom:50%;" />

<img src="C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241012111051816.png" alt="image-20241012111051816" style="zoom:50%;" />

### 向量链式法则

<img src="C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241012111435505.png" alt="image-20241012111435505" style="zoom:50%;" />

### 例子：

<img src="C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241012111720230.png" alt="image-20241012111720230" style="zoom:50%;" />

### 计算图：

<img src="C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241012111938507.png" alt="image-20241012111938507" style="zoom:50%;" />

<img src="C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241012112015786.png" alt="image-20241012112015786" style="zoom:50%;" />

<img src="C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241012112048807.png" alt="image-20241012112048807" style="zoom:50%;" />

### 自动求导的两种模式：

<img src="C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241012112129861.png" alt="image-20241012112129861" style="zoom:50%;" />

<img src="C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241012112322115.png" alt="image-20241012112322115" style="zoom:50%;" />

<img src="C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241012112347750.png" alt="image-20241012112347750" style="zoom:50%;" />

### 复杂度：

<img src="C:/Users/WDMX/AppData/Roaming/Typora/typora-user-images/image-20241012112452701.png" alt="image-20241012112452701" style="zoom:50%;" />



































