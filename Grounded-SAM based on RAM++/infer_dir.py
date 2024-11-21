import argparse  # 导入argparse模块，用于命令行参数解析
import numpy as np  # 导入numpy库，用于数值计算
import random, os, time  # 导入random, os, time模块，分别用于随机操作、文件系统操作和时间计算

import torch  # 导入PyTorch库，用于深度学习模型的构建和训练

from PIL import Image  # 从PIL库导入Image模块，用于图像处理
from ram.models import ram_plus  # 从ram.models模块导入ram_plus模型，用于推理
from ram import inference_ram as inference  # 从ram模块导入inference_ram并重命名为inference，用于进行模型推理
from ram import get_transform  # 从ram模块导入get_transform，用于图像预处理

# 创建ArgumentParser对象，用于解析命令行参数
parser = argparse.ArgumentParser(
    description='Tag2Text inference for tagging and captioning')

# 添加预训练模型路径参数
parser.add_argument('--pretrained',
                    metavar='DIR',
                    help='path to pretrained model',
                    default='pretrained/ram_plus_swin_large_14m.pth')

# 添加图像尺寸参数
parser.add_argument('--image-size',
                    default=384,
                    type=int,
                    metavar='N',
                    help='input image size (default: 448)')

if __name__ == "__main__":
    # 解析命令行参数
    args = parser.parse_args()

    # 设置设备为GPU（如果可用），否则为CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 获取图像预处理函数
    transform = get_transform(image_size=args.image_size)

    # 加载模型，设置为评估模式，并将其移动到指定设备
    model = ram_plus(pretrained=args.pretrained,
                     image_size=args.image_size,
                     threshold=0.8,
                     vit='swin_l')
    model.eval()
    model = model.to(device)

    # 设置图片文件的扩展名和根目录
    # C:\Users\WDMX\Desktop\recognize-anything-main\datasets\LZCimages
    prex = '.jpg'
    root = r'C:\Users\WDMX\Desktop\recognize-anything-main\datasets\LZCimages'

    # 获取所有符合扩展名的文件路径
    flist = [os.path.join(root, x) for x in os.listdir(root) if prex in x]

    # 遍历文件列表，进行推理
    for path in flist:
        # 读取并预处理图像，然后移动到指定设备
        image = transform(Image.open(path)).unsqueeze(0).to(device)

        # 记录时间并进行推理
        t0 = time.time()
        res = inference(image, model)
        rt = time.time() - t0

        # 打印运行时间和推理结果
        print(f"运行时间：{rt:.4f}, 图片路径：{path}")
        print("Image Tags: ", res[0])
        print("图像标签: ", res[1])