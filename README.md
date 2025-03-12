# Deep Learning
## 深度学习汇总（Google Colab/本机GPU运行 - pytorch版）

### 学习推荐和内容描述：

## 一、B站UP主“跟李沐学AI”的《动手学深度学习》系列视频教程

【完结】动手学深度学习 PyTorch版 - https://space.bilibili.com/1567748478/channel/seriesdetail?sid=358497

## 二、B站UP主“3Blue1Brown”中国官方账号的《深度学习 Deep Learnig》系列视频教程

深度学习 Deep Learning - https://space.bilibili.com/88461692/channel/seriesdetail?sid=1528929

一共六集，📚3Blue1Brown 深度学习课程笔记见下文件夹《3b1b Deep Learning》：

【01】深度学习之神经网络结构解析

【02】深度学习之梯度下降法深度解析

## 三、深度学习电子书资源

《动手学深度学习》中文版本地址：https://zh-v2.d2l.ai/

《Dive into Deep Learning》英文版地址：https://d2l.ai/ 

## 四、Deep Learning Review - 深度学习综述论文

英文版和中文版（自译）都放在 Task4 的文件包中了（Deep Learning Learner必看）

其中的《深度学习综述论文总结》是我对论文的一个小小的总结，而知识点GPT理解是对其中的部分知识点的细致理解，便于新手读者快速理解知识点含义

## 五、哈利波特主要人物图像识别（HP_image_recognition）

1、图片爬取程序 HP_image_collection 和 Google_HP_image_collection（本程序数据集：hpdata）

2、包含人物 Harry Potter、Hermione Granger、Ron Weasley、Albus Dumbledore、Rubeus Hagrid、Severus Snape、Voldemort

3、通过HP_image_collection爬取人均100张彩色图片（自采集补充），训练预测比为9:1，基于Pytorch第三方库、数据增强与标准化技术、预训练的 ResNet50 模型和作者本机4060GPU，使用Jupyter NoteBook进行代码实现，并对预测图片进行可视化操作。训练结果为train_loss：0.2905，accuracy：0.9006；test_loss：0.0330，accuracy：0.9939。望各位朋友对哈利波特人物识别库进行扩充！

## 六、Pytorch-CUDA GPU安装教程和“Friend_recognition”

1、Pytorch-CUDA GPU安装教程（image_tutorial&tutorial_notes）

相关网站：Pytorch-GPU：https://pytorch.org/

cuDNN：https://developer.nvidia.com/rdp/cudnn-archive

CUDA Toolkit：https://developer.nvidia.com/cuda-12-3-0-download-archive?target_os=Windows

2、Friend_recognition：基于本人及两位好友的数百张图片集，训练的简单卷积神经网络，用于入门学习和训练流程理解，使用展示图像的可视化辅助函数帮助理解

## 七、Deep learning Studying Plan/Task

### 【流程学习任务】DL-Task 1 to DL-Task 4

### Task 1 - Task 4

1、概述		

•	熟悉机器学习、深度学习基础知识，熟练使用google colab.
 
•	初步学习和了解深度学习库pytorch, 阅读并总结发表在Nature上的深度学习综述论文.

2、成果

•	《深度学习编程环境安装报告》

•	《深度学习库pytorch实验报告》

•	《深度学习综述论文总结》

3、内容

•	学习google colab, https://colab.research.google.com/ 完成google colab基本教程的学习，从使用入门到机器学习示例。
 
•	学习《动手学深度学习》“序言”，“安装”，“符号”。代码版本用pytorch版。并按照“安装”一节的教程，安装 Miniconda，安装深度学习框架pytorch和d2l软件包，下载 D2L Notebook。

•	学习《动手学深度学习》（《Dive into Deep Learning》）第一至四章（从Introduction到Linear Neural Networks for Classification），并在colab上运行代码。代码版本用pytorch版。英文版地址：https://d2l.ai/。
若英文阅读有难度，可参考中文版进行理解，但代码务必使用英文版上的。 中文版本地址：https://zh-v2.d2l.ai/

4、熟悉深度学习库pytorch，完成入门教程的学习和理解，并在colab运行如下示例，地址为https://pytorch.org/tutorials/

Introduction to PyTorch[ - ]

•	Learn the Basics、Quickstart、Tensors、Datasets & Dataloaders、Transforms、Build the Neural Network、Automatic Differentiation with torch.autograd、Optimizing Model Parameters、Save and Load the Model

Learning PyTorch[ - ]

•	Deep Learning with PyTorch: A 60 Minute Blitz、Learning PyTorch with Examples、What is torch.nn really?、Visualizing Models, Data, and Training with TensorBoard

5、阅读并总结发表在Nature上的深度学习综述论文

https://www.nature.com/articles/nature14539

#### Task 5

1、概述		

卷积神经网络；熟悉pytorch库，完成入门教程的学习，进行图像分类等基本任务。

2、成果

《基于CNN的图像分类实验报告》

3、内容

•	学习《动手学深度学习》第五至六章，“4. 多层感知机；5. 深度学习计算；6. 卷积神经网络；”，并在colab上运行代码。https://zh-v2.d2l.ai/

•	使用pytorch进行图像分类等基本任务 https://pytorch.org/tutorials/

Image and Video[ - ]

Transfer Learning for Computer Vision Tutorial

#### Task 6

1、概述		

现代卷积神经网络、循环神经网络基础。

2、成果

基于Resnet的图像分类实验报告

3、内容

•	学习《动手学深度学习》第七章，第八章： 7. 现代卷积神经网络；8. 循环神经网络”，在colab上运行pytorch版本代码,并理解。https://zh-v2.d2l.ai/

•	完成第七章练习2，“参考ResNet论文 [He et al., 2016a]中的表1，以实现不同的变体。”，并在Fashion-MNIST数据集上训练你实现的ResNet变体，形成实验报告。

#### Task 7

1、概述		

循环神经网络与注意力机制

2、成果

《基于RNN的机器翻译实验报告》

3、内容

•	学习《动手学深度学习》第8章至第10章： 8. 循环神经网络, 9现代循环神经网络；10. 注意力机制”，在colab上运行pytorch版本代码,并理解。https://zh-v2.d2l.ai/

•	完成9.5-9.7节的机器翻译模型训练和测试，并形成《基于RNN的机器翻译实验报告》。

#### Task 8

1、概述		

理解常见的目标检测、语义分割方法原理，调通相应程序，并完成目标检测实践任务

2、成果

《基于深度学习的目标检测实验报告》

3、内容

•	学习《动手学深度学习》第十三章“13. 计算机视觉”，在colab上运行pytorch版本代码，并深入理解。理解常见的目标检测、语义分割方法原理。

•	使用pytorch进行目标检测任务 https://pytorch.org/tutorials/

Image and Video[ - ] TorchVision Object Detection Finetuning Tutorial

#### Task 9

探索性任务：大模型驱动的开放世界目标感知：融合使用多个大模型的多模态感知能力，实现开放世界零样本条件下，自动目标检测、识别、分割功能。

## 八、大模型驱动的开放世界目标感知（Grounded-SAM based on RAM++）

•  算法训练：学习深度学习和神经网络算法，使用 PyTorch 训练图像分类器模型；

•  集成模型：融合使用多个大模型的多模态感知能力，实现开放世界零样本条件下，自动目标检测、识别、分割功能,使用 RAM++ 模型检测和识别物体，生成物体名称列表，设计物体分割计划并创建相应字段，调用 Grounded-SAM 分解并生成结果；

•  性能评估：使用 OCID 数据集测试模型并达到高精度（Mean IoU > 0.975；精度 1.0 和召回率 1.0）；

•  持续研究：集成模型拓展以推进无人平台的环境感知研究，预计持续研究和应用的推进，最终将提供实用有效的解决方案，释放研究目标的自主性，使研究目标能够在动态和不可预测的环境中执行复杂任务，并安全可靠地与人类协作。



