# ViT-Diff-DCA 泛化性实验

本目录用于验证 `Diff Attention + DCA` 从 Swin Transformer 迁移到 ViT-B/16 后的泛化性。

## 文件说明

- `vit_diff_dca_model.py`: ViT-B/16 上的 Diff Attention 与 DCA 注入逻辑。
- `vit_train.py`: ViT-B/16 baseline 训练脚本。
- `vit_diff_train.py`: ViT-B/16 + Diff Attention 消融训练脚本。
- `vit_dca_train.py`: ViT-B/16 + DCA 消融训练脚本。
- `vit_diff_dca_train.py`: ViT-Diff-DCA 训练脚本，核心训练参数与 `vit_train.py` 对齐。
- `../../evaluate/evaluate_vit_diff.py`: ViT-Diff 测试集评估脚本。
- `../../evaluate/evaluate_vit_dca.py`: ViT-DCA 测试集评估脚本。
- `../../evaluate/evaluate_vit_diff_dca.py`: ViT-Diff-DCA 测试集评估脚本。

## 对齐参数

该实验沿用 ViT baseline 的主要设置：

- `IMAGE_SIZE = 224`
- `BATCH_SIZE = 32`
- `NUM_EPOCHS = 30`
- `LR_BACKBONE = 1e-5`
- `LR_NEW = 3e-4`
- `WEIGHT_DECAY = 0.05`
- `LABEL_SMOOTHING = 0.05`
- `WARMUP_EPOCHS = 5`
- `FREEZE_EPOCHS = 3`
- 训练增强：`Resize`、随机水平翻转、随机垂直翻转、随机旋转、ImageNet 归一化
- 验证/测试增强：`Resize`、ImageNet 归一化

## 建议运行

```bash
python classification/train/vit/vit_diff_dca_train.py
python classification/train/vit/vit_diff_train.py
python classification/train/vit/vit_dca_train.py
python classification/evaluate/evaluate_vit_diff.py
python classification/evaluate/evaluate_vit_dca.py
python classification/evaluate/evaluate_vit_diff_dca.py
```

训练输出默认保存在：

```text
result/checkpoints/
```
