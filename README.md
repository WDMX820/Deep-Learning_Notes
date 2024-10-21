# Deep Learning
深度学习汇总（Google Colab/本机GPU运行 - pytorch版）

学习推荐和内容描述：

1、B站UP主“跟李沐学AI”的《动手学深度学习》系列视频教程

【完结】动手学深度学习 PyTorch版 - https://space.bilibili.com/1567748478/channel/seriesdetail?sid=358497

2、B站UP主“3Blue1Brown”中国官方账号的《深度学习 Deep Learnig》系列视频教程

深度学习 Deep Learning - https://space.bilibili.com/88461692/channel/seriesdetail?sid=1528929

3、深度学习电子书资源

《动手学深度学习》中文版本地址：https://zh-v2.d2l.ai/

《Dive into Deep Learning》英文版地址：https://d2l.ai/ 

4、Deep Learning Review - 深度学习综述论文

英文版和中文版（自译）都放在 Task4 的文件包中了（Deep Learning Learner必看）

其中的《深度学习综述论文总结》是我对论文的一个小小的总结，而知识点GPT理解是对其中的部分知识点的细致理解，便于新手读者快速理解知识点含义

5、哈利波特主要人物图像识别（HP_image_recognition）

图片爬取程序：HP_image_collection 和 Google_HP_image_collection（本程序数据集：hpdata）

目前包含人物：Harry Potter、Hermione Granger、Ron Weasley、Albus Dumbledore、Rubeus Hagrid、Severus Snape、Voldemort

通过HP_image_collection爬取人均100张彩色图片（自采集补充），训练预测比为9:1，基于Pytorch第三方库、数据增强与标准化技术、预训练的 ResNet50 模型和作者本机4060GPU，使用Jupyter NoteBook进行代码实现，并对预测图片进行可视化操作。

训练结果：train_loss为0.2905，accuracy为0.9006；test_loss为0.0330，accuracy为0.9939。望各位朋友对哈利波特人物识别库进行扩充！



