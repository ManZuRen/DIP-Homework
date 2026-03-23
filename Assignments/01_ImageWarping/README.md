# Assignment 01: Image Warping

本仓库是中国科学技术大学数字图像处理课程第一次作业 `Assignment_01` 的代码实现与实验说明。

本次作业完成了两部分内容：

1. 基础图像几何变换：缩放、旋转、平移、水平翻转。
2. 基于控制点的图像形变：使用 Moving Least Squares (MLS) 仿射形变实现交互式局部变形。

## Project Overview

本项目提供两个基于 Gradio 的交互程序，用户可直接在网页端上传图像、调整参数或点击控制点，并实时查看变换结果。

- `run_global_transform.py`
  用于基础几何变换。
- `run_point_transform.py`
  用于控制点引导的局部图像形变。

## Requirements

建议使用 Python 3.9 及以上版本。

安装依赖：

```bash
python -m pip install -r requirements.txt
```

依赖列表如下：

- `numpy`
- `opencv-python`
- `gradio`

## Running Scripts

### 1. Global Transformation

运行：

```bash
python run_global_transform.py
```

程序启动后会打开一个本地 Gradio 页面，用于完成以下操作：

- `Scale`：按图像中心缩放。
- `Rotation`：按图像中心旋转。
- `Translation X / Y`：在变换后继续进行平移。
- `Flip Horizontal`：执行水平翻转。

实现要点：

- 先对白底画布进行扩边，减少旋转和缩放时的边界裁切。
- 将缩放、旋转、翻转和平移统一表示为齐次坐标下的矩阵乘法。
- 最终通过 `cv2.warpAffine` 完成仿射重采样。

### 2. Point-Guided Warping

运行：

```bash
python run_point_transform.py
```

使用流程：

1. 上传一张待处理图像。
2. 在图像上按“源点、目标点、源点、目标点”的顺序依次点击。
3. 点击 `Run Warping` 生成形变结果。
4. 点击 `Clear Points` 清空控制点并重新选择。

实现要点：

- 采用 MLS Affine Deformation。
- 对输出图像中的每个像素执行反向映射，根据控制点局部估计仿射变换。
- 使用距离倒数构造权重，距离越近的控制点影响越大。
- 最终通过 `cv2.remap` 完成无空洞采样。

## Input and Output

### Global Transformation

输入：

- 一张彩色图像。
- 缩放系数、旋转角度、平移参数、翻转选项。

输出：

- 经过组合几何变换后的图像。

### Point-Guided Warping

输入：

- 一张彩色图像。
- 至少 3 对控制点，每对由一个源点和一个目标点组成。

输出：

- 根据控制点对应关系生成的局部形变结果图像。

## Results

### Result 1: Global Transformation

下图展示了基础几何变换模块的运行结果。

<img src="pics/p1.png" alt="Global transformation result" width="800">

结果分析：

- 图像能够正确完成缩放、旋转、平移与水平翻转。
- 通过先扩边再变换，明显减轻了旋转后图像内容被裁切的问题。
- 仿射矩阵组合方式清晰，适合扩展更多全局变换操作。

### Result 2: Point-Guided Warping

下图展示了控制点引导形变结果。

<img src="pics/p2.png" alt="Point-guided warping result" width="800">

结果分析：

- 控制点附近区域能随目标点移动产生局部弹性变形。
- MLS 方法能够保持整体过渡较平滑，避免出现大面积断裂。
- 当控制点数量较少或分布不均匀时，远离控制点的区域形变效果会相对弱一些。

### Demo Video

实验演示视频预览如下：

<img src="pics/video1_preview.gif" alt="Demo preview" width="800">

原始视频链接：

- [video1.mp4](pics/video1.mp4)

## Discussion

本次作业验证了两类典型图像变换方法：

- 全局几何变换适用于整体结构一致的变换任务，实现简单，计算代价低。
- 基于控制点的 MLS 形变更适合局部编辑和非刚性变形，但计算量明显更大。

从实验结果看，基础变换模块能够稳定输出正确结果；控制点形变模块在交互体验和视觉连续性方面表现较好，但在控制点过少时对复杂变形的表达能力有限。后续若进一步优化，可以考虑：

- 引入更高效的向量化或并行计算以减少逐像素 MLS 的耗时。
- 增加控制点编辑、删除和保存功能，提升交互性。
- 对不同 MLS 变体进行比较，如 similarity 或 rigid deformation。

## Work Division

本项目为个人独立完成，不涉及合作分工。

## File Structure

- `run_global_transform.py`：基础几何变换程序。
- `run_point_transform.py`：控制点引导图像形变程序。
- `requirements.txt`：运行依赖。
- `pics/p1.png`：基础几何变换结果图。
- `pics/p2.png`：控制点形变结果图。
- `pics/video1_preview.gif`：演示视频预览。
- `pics/video1.mp4`：实验演示视频。

## References

本项目参考了以下资料与来源：

1. 课程作业要求与课堂相关内容。
2. Schaefer S, McPhail T, Warren J. Image deformation using moving least squares. ACM Transactions on Graphics, 2006.
3. OpenCV 官方文档：仿射变换与图像重映射相关接口说明。
4. Gradio 官方文档：交互式网页界面构建方法。

