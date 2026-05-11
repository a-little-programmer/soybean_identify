# AGENTS.md

本仓库主要用于大豆品种细粒度视觉分类实验，核心代码集中在 `resnet/` 目录。后续代理在修改代码、补实验脚本或生成论文辅助材料时，必须优先保持现有实验主线、命名习惯和训练流程的一致性。

## 数据处理链路

- 本项目不是直接对整张原始图像做品种分类，而是采用两阶段视觉流程。
- 第一阶段使用 YOLO 目标检测定位原始图像中的大豆籽粒或目标区域，用于从复杂原图中获得候选豆粒位置。
- 第二阶段基于检测结果进行人工校验与手动裁剪，形成面向分类任务的单粒或局部大豆图像数据。
- `resnet/` 目录中的模型训练与评估脚本默认处理的是已经裁剪、清洗并按类别组织好的分类数据集，而不是 YOLO 检测原图。
- 论文或报告中描述系统流程时，应写成“YOLO 目标检测定位 + 人工校验裁剪 + 细粒度分类模型识别”的级联流程，不要把分类模型描述成直接完成整图检测和分类。
- 如果后续新增端到端脚本，应明确区分检测阶段、裁剪阶段和分类阶段的输入输出，避免混用检测指标与分类指标。

## 图像预处理与数据增强

- 分类阶段使用的是经过裁剪与类别整理后的图像数据，数据增强是训练管道的重要组成部分。
- 训练集通常使用 `Resize`、随机水平翻转、随机垂直翻转、随机旋转和 ImageNet 均值方差归一化。
- 验证集和测试集必须使用确定性变换，通常只做 `Resize` 和归一化，不要使用随机增强。
- 如果使用 `RandomResizedCrop`、`ColorJitter`、HSV 增强或其他更强增强策略，必须把它视为新的实验变量，并在消融或实验说明中单独交代。
- 不要在验证集或测试集上使用随机裁剪、随机翻转、随机旋转等增强，否则评估结果不可复现。
- 论文或报告中描述数据处理时，应同时说明检测裁剪流程和分类训练阶段的数据增强流程，避免只写模型结构而忽略数据管道贡献。

## 项目主线

- 当前主要研究对象是大豆品种细粒度分类，不要把任务泛化成普通图像分类。
- 当前核心模型路线是 `Swin Transformer + Diff Attention + DCA`，对应文件优先参考 `resnet/swin_diff_dca_train.py` 和 `resnet/evaluate_swin_diff_dca.py`。
- `Diff Attention` 当前应理解为注入到 Swin 深层窗口注意力中的差分注意力分支，主要作用在 Stage 4。
- `DCA` 当前应理解为 Stage 3 内部的动态残差通道路由，用 anchor block 特征辅助 target block。
- 不要把未正式落地或未完成消融的方案写成已完成成果，例如 `ArcFace`、`Top-K token branch`、显式 `Stage3+Stage4 fusion`、`WeightedRandomSampler`。

## 代码风格

- Python 文件保留 `# -*- coding: utf-8 -*-`。
- 保持现有脚本式实验风格：配置常量放在文件顶部，随后是模块定义、数据加载、优化器/调度器、训练或评估主流程。
- 训练脚本优先使用清晰的全局配置名，例如 `DATA_DIR`、`SAVE_DIR`、`MODEL_NAME`、`CLASS_INDEX_NAME`、`MODEL_ID`、`IMAGE_SIZE`、`BATCH_SIZE`、`NUM_EPOCHS`。
- 路径优先使用 `BASE_DIR = os.path.dirname(os.path.abspath(__file__))`，再基于 `BASE_DIR` 拼接数据、权重和输出目录。不要新增依赖当前工作目录的相对路径。
- 保持现有训练日志风格，可以使用中文提示，但不要加入夸张、营销式或不可验证的描述。
- 新增复杂模块时，注释说明它解决什么问题、注入到哪个 stage、是否改变预训练权重加载方式。
- 不要大规模重构已有训练脚本，除非用户明确要求。优先做小范围、可验证的修改。

## 实验公平性

- 做 baseline 或消融实验时，必须对齐关键超参：`MODEL_ID`、`IMAGE_SIZE`、`BATCH_SIZE`、`NUM_EPOCHS`、`WEIGHT_DECAY`、`LABEL_SMOOTHING`、数据增强、冻结策略、学习率调度。
- 以 `Macro-F1` 作为长尾细粒度分类的核心选择指标；`Accuracy` 和 `Weighted-F1` 只能作为辅助指标。
- `f1_score` 必须显式传入完整类别列表：

```python
f1_score(labels_all, preds_all, labels=list(range(num_classes)), average="macro", zero_division=0)
```

- 保存 checkpoint 时同步保存类别映射 JSON，文件名要与模型版本对应。
- 评估脚本必须与训练脚本的模型结构 1:1 对齐。凡是训练中注入了 `Diff`、`DCA` 或其他自定义模块，评估端必须先构建同样结构再加载权重。

## 训练规范

- 固定随机种子时同时设置 `random`、`numpy`、`torch`、`torch.cuda`，并设置：

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

- 使用 AMP 时优先使用 PyTorch 2.x 写法：

```python
torch.amp.autocast("cuda", enabled=USE_AMP)
torch.amp.GradScaler("cuda", enabled=USE_AMP)
```

- 反向传播中如果使用梯度裁剪，必须先 `scaler.unscale_(optimizer)` 再裁剪。
- 冻结预热阶段要同时考虑两件事：参数 `requires_grad` 和模型 `train/eval` 状态。冻结 backbone 时应关闭 DropPath/Dropout，只让新模块和分类头保持训练态。
- 分组学习率应明确区分 backbone 参数和新模块参数。新模块关键词应集中在函数中维护，例如 `get_fast_keywords()`。

## 长尾处理

- 当前代码中真实落地的是类别加权损失，通常基于 `1 / sqrt(class_count)`。
- 如果新增 `WeightedRandomSampler`，必须明确说明它是新实验变量，并与类别加权损失分别做消融，避免过补偿。
- 不要在论文或报告中声称同时使用采样和加权损失，除非对应训练脚本确实同时启用了两者。

## 模型修改边界

- 修改 Swin 注意力时，优先使用实例级注入或局部替换，谨慎使用全局 monkey patch。
- 如果使用全局 monkey patch，例如替换 `timm.models.swin_transformer.WindowAttention`，必须确保脚本是独立运行场景，并在注释中说明副作用。
- 自定义注意力模块必须正确处理 `relative_position_index`。该 buffer 通常不是持久化 state_dict 项，实例级替换时必须显式拷贝。
- 保留预训练权重是优先目标。新增模块应尽量以残差或零初始化方式接入，避免无意中重建原始 `qkv`、`proj`、`fc1`、`fc2` 等已预训练权重。

## 评估与论文材料

- 每个最终模型都应配套评估脚本，至少输出 per-class precision、recall、F1、support、混淆矩阵和总体指标。
- 对论文有价值的分析包括：混淆矩阵、困难类别 bad case、Grad-CAM 或 attention 可视化、参数量、FLOPs、模型大小和单张推理时间。
- 写论文或简历时，只陈述代码和实验已经真实支持的内容。未完成消融的设计可以写成“探索”或“设计”，不要写成最终贡献。
- 困难类别分析应优先关注低 F1 或高混淆类别，例如 `zd51`、`zd53`、`zd57`、`zd59`、`zd61`、`sd29`、`xd18`。

## 文件命名

- 训练脚本使用 `<model>_train.py` 或 `<variant>_train.py`。
- 评估脚本使用 `evaluate_<variant>.py`。
- checkpoint 使用 `best_<variant>.pth`。
- 类别映射使用 `class_indices_<variant>.json`。
- 输出目录或日志名应包含模型变体，避免覆盖已有实验结果。

## 运行与依赖

- 主要依赖包括 `torch`、`torchvision`、`timm`、`numpy`、`scikit-learn`、`tqdm`、`matplotlib`、`opencv-python`、`grad-cam`。
- 服务器环境可能无法稳定访问 Hugging Face。新增依赖预训练权重的脚本时，应提供本地权重或缓存兜底逻辑。
- 不要默认真实数据路径在本地存在。需要 smoke test 时可使用随机 tensor 验证模型构建、前向、反向和优化器步骤。
- 用户的训练与评估代码主要在云端服务器运行。本地代理修改训练或评估脚本时，只负责直接改文件并向用户说明改动内容；不要在本地运行训练、评估、权重加载或依赖云端数据/环境的命令，除非用户明确要求。
- 如果需要验证云端脚本，应把修改后的文件路径、关键改动点和建议运行命令发给用户，由用户在云端执行。

## 禁止事项

- 不要把实验主线从 `swin_diff_dca` 随意切换到新模型，除非用户明确要求。
- 不要删除或覆盖已有 checkpoint、日志、评估报告。
- 不要把路径硬编码成某台服务器的绝对路径。
- 不要在没有消融结果的情况下声称某个模块“显著提升”。
- 不要把未实际启用的技术写进项目成果。
