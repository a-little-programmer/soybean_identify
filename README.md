# Soybean Fine-Grained Visual Classification

本项目用于大豆品种细粒度视觉分类实验，整体流程采用级联式视觉方案：

1. 使用 YOLO 对原始图像中的大豆籽粒或目标区域进行定位。
2. 基于检测结果进行人工校验与手动裁剪，得到分类阶段使用的单粒或局部图像。
3. 使用分类模型对已裁剪、清洗并按类别组织的数据进行品种识别。

当前主要实验代码集中在 `resnet/` 目录，核心路线为 `Swin Transformer + Diff Attention + DCA`：

- `resnet/swin_diff_dca_train.py`：主模型训练脚本。
- `resnet/evaluate_swin_diff_dca.py`：与主模型结构对齐的评估脚本。
- `resnet/measure_models.py`：模型参数量、复杂度和推理开销统计相关脚本。
- `train.py`：YOLO 检测阶段训练入口。

## 环境依赖

主要依赖包括：

- `torch`
- `torchvision`
- `timm`
- `numpy`
- `scikit-learn`
- `tqdm`
- `matplotlib`
- `opencv-python`
- `grad-cam`
- `ultralytics`

## 数据说明

分类训练脚本默认处理已经裁剪、清洗并按类别组织好的分类数据集，不直接对整张原始图像做品种分类。验证集和测试集应使用确定性变换，避免随机裁剪、随机翻转或随机旋转影响复现。

## 权重文件

模型权重通常较大，不建议直接提交到 GitHub。本仓库默认通过 `.gitignore` 排除 `*.pth`、`*.pt`、`*.ckpt`、`*.onnx` 和 `*.weights` 文件，仅保留轻量的类别映射 JSON。
