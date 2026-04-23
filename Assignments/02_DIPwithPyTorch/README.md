<img width="1545" height="743" alt="image" src="https://github.com/user-attachments/assets/87eaf426-138a-442c-a88e-c7e0dbcdd868" />
# Assignment 2 Report - DIP with PyTorch

本报告对应数字图像处理课程作业 2，包含两个部分：

1. Poisson Image Editing
2. FCN-based Image-to-Image Translation
## Overview

本次作业目标如下：

1. 在 PyTorch 框架下完成 Poisson Image Editing 的关键补全，并实现交互式图像融合。
2. 在 `Pix2Pix/FCN_network.py` 中实现 Fully Convolutional Network，并在 Facades 数据集上完成图像到图像翻译训练。

代码目录如下：

- `run_blending_gradio.py`：任务一主程序
- `Pix2Pix/FCN_network.py`：任务二网络结构
- `Pix2Pix/train.py`：任务二训练脚本
- `Pix2Pix/facades_dataset.py`：任务二数据读取
- `Pix2Pix/datasets/facades/`：任务二数据集

## Environment

本作业在 Windows + PowerShell 环境下完成，主要依赖如下：

- Python 3.10
- PyTorch
- NumPy
- Pillow
- Gradio
- OpenCV (`opencv-python`)

实验使用的 Python 解释器为：

```powershell
C:\Users\Admin\Desktop\vmesh\.venv\Scripts\python.exe
```

项目根目录：

```powershell
DIP\DIP-Homework\Assignments\02_DIPwithPyTorch
```

## Task 1 - Poisson Image Editing

### Task Description

任务一要求在 `run_blending_gradio.py` 中补全以下核心部分：

- `create_mask_from_points`
- `cal_laplacian_loss`

整体目标是在前景图中选取一个多边形区域，将其移动到背景图指定位置，并通过梯度域约束得到更自然的融合结果。

### Method

任务一的处理流程如下：

1. 在前景图像中手动点击得到 polygon 顶点。
2. 根据 polygon 生成前景 mask。
3. 根据界面中的 `dx, dy` 将 polygon 平移到背景图中，并生成背景 mask。
4. 初始化待优化的融合图像 `blended_img`。
5. 使用离散拉普拉斯算子计算结构差异，并在 mask 区域内构造 loss。
6. 通过迭代优化得到最终融合结果。

本实验使用的离散拉普拉斯核为：

```text
0  1  0
1 -4  1
0  1  0
```

### Key Implementation

#### 1. Polygon to Mask

使用 `PIL.ImageDraw.Draw.polygon` 将用户圈选的多边形区域填充为二值 mask：

- 区域内像素值为 `255`
- 区域外像素值为 `0`

#### 2. Laplacian Loss

分别对前景图和当前融合图进行拉普拉斯卷积，并仅在选中区域内取值后计算均方误差：

```text
L = mean((Laplace(source) - Laplace(blended))^2)
```

该损失的作用是让融合区域在局部结构上尽量保持前景图中的梯度特征。

### How to Run

```powershell
cd C:\Users\Admin\Desktop\DIP\DIP-Homework\Assignments\02_DIPwithPyTorch
C:\Users\Admin\Desktop\vmesh\.venv\Scripts\python.exe run_blending_gradio.py
```

启动后在浏览器中打开：

```text
http://127.0.0.1:7860
```
### Results
<img width="1545" height="743" alt="image" src="https://github.com/user-attachments/assets/2d27a09b-d861-4b6a-9b14-38805ab96d0e" />
<img width="1603" height="620" alt="image-1" src="https://github.com/user-attachments/assets/af95a7eb-7056-4780-b597-a9262b68ee0a" />



### Analysis

- 与直接复制像素相比，Poisson Image Editing 更强调梯度信息的一致性，因此边界过渡更自然。
- 本实现基于训练优化，并非解方程，因此在 CPU 上运行时耗时较长。
- 结果质量与选区形状、目标位置和优化迭代过程都有较强关系。

## Task 2 - FCN-based Image-to-Image Translation

### Task Description

任务二要求在 `Pix2Pix/FCN_network.py` 中实现 Fully Convolutional Network，并使用 Facades 数据集完成训练。

根据题目说明，需要完成以下工作：

1. 补全 `FCN_network.py`
2. 下载并组织 Facades 数据集
3. 运行 `train.py` 完成训练

### Dataset

本实验使用的数据集为 Facades，目录如下：

```text
Pix2Pix/datasets/facades/
```

数据划分如下：

- `train/`
- `val/`
- `test/`

本地生成的数据列表文件如下：

- `Pix2Pix/train_list.txt`
- `Pix2Pix/val_list.txt`

本次实验中：

- 训练集数量：400
- 验证集数量：100

### Network Architecture

本次实现的是一个对称的全卷积编码器-解码器网络，属于 FCN baseline，而不是完整的 GAN 版 pix2pix。

编码器通道变化：

- `3 -> 64`
- `64 -> 128`
- `128 -> 256`
- `256 -> 512`

解码器通道变化：

- `512 -> 256`
- `256 -> 128`
- `128 -> 64`
- `64 -> 3`

网络的空间尺寸变化如下：

```text
256 -> 128 -> 64 -> 32 -> 16 -> 32 -> 64 -> 128 -> 256
```

主要设计如下：

- 下采样层：`Conv2d(kernel_size=4, stride=2, padding=1)`
- 上采样层：`ConvTranspose2d(kernel_size=4, stride=2, padding=1)`
- 中间使用 `BatchNorm + ReLU`
- 输出层使用 `Tanh()`

### Training

训练命令如下：

```powershell
cd C:\Users\Admin\Desktop\DIP\DIP-Homework\Assignments\02_DIPwithPyTorch\Pix2Pix
C:\Users\Admin\Desktop\vmesh\.venv\Scripts\python.exe train.py
```

训练配置如下：

- 优化器：Adam
- 学习率：`0.001`
- `betas=(0.5, 0.999)`
- 损失函数：L1 Loss
- 学习率调度：StepLR
- 训练轮数：300 epochs

### Checkpoints and Outputs

训练脚本会自动保存以下结果：

- `Pix2Pix/checkpoints/`
- `Pix2Pix/train_results/`
- `Pix2Pix/val_results/`

本次训练产生的权重包括：

- `Pix2Pix/checkpoints/pix2pix_model_epoch_50.pth`
- `Pix2Pix/checkpoints/pix2pix_model_epoch_100.pth`
- `Pix2Pix/checkpoints/pix2pix_model_epoch_150.pth`
- `Pix2Pix/checkpoints/pix2pix_model_epoch_200.pth`
- `Pix2Pix/checkpoints/pix2pix_model_epoch_250.pth`
- `Pix2Pix/checkpoints/pix2pix_model_epoch_300.pth`

最终模型可写为：

```text
Pix2Pix/checkpoints/pix2pix_model_epoch_300.pth
```

### Results
训练集结果

<img width="768" height="256" alt="result_4" src="https://github.com/user-attachments/assets/e74d3dbe-563f-46c4-804e-7971f0d2e465" />
<img width="768" height="256" alt="image-2" src="https://github.com/user-attachments/assets/1563e836-d66b-4b54-b4b3-0898c4b5be5d" />

验证集结果

<img width="768" height="256" alt="result_5" src="https://github.com/user-attachments/assets/36ed7773-69dd-4eca-b17a-0de01ec34edd" />
<img width="768" height="256" alt="result_4" src="https://github.com/user-attachments/assets/52c7d2bc-d997-4b47-8320-ba4e4c7ae199" />


### Analysis

- 当前实现是 FCN 图像翻译基线，不包含判别器，因此不属于完整的 pix2pix GAN。
- 模型通过 L1 loss 学习输入图像到目标图像之间的映射关系。
- 该结构训练稳定、实现简单，但是表达能力不足，会出现过拟合，在验证集上表现不佳。

## Conclusion

本次作业完成了传统图像融合与基于深度学习的图像翻译两个部分：

1. 任务一实现了 Poisson Image Editing 的关键步骤，并通过 Gradio 实现交互式融合。
2. 任务二实现了 Fully Convolutional Network，并在 Facades 数据集上完成训练与结果保存。

## References

### Task 1

- Perez, P., Gangnet, M., and Blake, A. *Poisson Image Editing*. ACM SIGGRAPH 2003.  
  Link: https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf
- PyTorch Documentation: `torch.nn.functional.conv2d`  
  Link: https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html

### Task 2

- Pix2Pix Project Page  
  Link: https://phillipi.github.io/pix2pix/
- Fully Convolutional Networks for Semantic Segmentation  
  Link: https://arxiv.org/abs/1411.4038
- Facades Dataset  
  Link: https://cmp.felk.cvut.cz/~tylecr1/facade/
- Pix2Pix Datasets  
  Link: https://github.com/phillipi/pix2pix#datasets
