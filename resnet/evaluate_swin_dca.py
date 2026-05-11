# -*- coding: utf-8 -*-
import os
import csv
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import timm

# ==============================================================================
# 0. 配置区域
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/classifier_dataset_hsv/test"))       
WEIGHT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_swin_dca_only.pth")    # 指向 DCA 消融实验的权重
EVAL_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(BASE_DIR, EVAL_NAME)
MODEL_ARCH = "swin_base_patch4_window7_224.ms_in22k_ft_in1k" 

BATCH_SIZE = 64 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INIT_DCA_GAMMA = 0.0

def save_evaluation_results(
    report_lines,
    results,
    summary,
    all_labels,
    all_preds,
    dataset,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    report_path = os.path.join(OUTPUT_DIR, "report.txt")
    per_class_path = os.path.join(OUTPUT_DIR, "per_class_metrics.csv")
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    predictions_path = os.path.join(OUTPUT_DIR, "predictions.csv")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    with open(per_class_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "support", "precision", "recall", "f1", "avg_loss"])
        writer.writeheader()
        for row in results:
            writer.writerow({
                "class": row["name"],
                "support": int(row["support"]),
                "precision": float(row["p"]),
                "recall": float(row["r"]),
                "f1": float(row["f1"]),
                "avg_loss": float(row["loss"]),
            })

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(predictions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "true_idx", "true_class", "pred_idx", "pred_class"])
        writer.writeheader()
        for idx, (label, pred) in enumerate(zip(all_labels, all_preds)):
            image_path = dataset.samples[idx][0]
            writer.writerow({
                "image_path": image_path,
                "true_idx": int(label),
                "true_class": dataset.classes[int(label)],
                "pred_idx": int(pred),
                "pred_class": dataset.classes[int(pred)],
            })

    print(f"💾 评估结果已保存至: {OUTPUT_DIR}")

def save_confusion_matrices(all_labels, all_preds, class_names):
    labels = list(range(len(class_names)))
    cm = confusion_matrix(all_labels, all_preds, labels=labels)
    cm_norm = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    np.savetxt(os.path.join(OUTPUT_DIR, "confusion_matrix_counts.csv"), cm, fmt="%d", delimiter=",")
    np.savetxt(os.path.join(OUTPUT_DIR, "confusion_matrix_normalized.csv"), cm_norm, fmt="%.6f", delimiter=",")

    for matrix, filename, title, fmt in [
        (cm, "confusion_matrix_counts.png", "Confusion Matrix (Counts)", "d"),
        (cm_norm, "confusion_matrix_normalized.png", "Confusion Matrix (Row Normalized)", ".2f"),
    ]:
        fig_size = max(10, len(class_names) * 0.55)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            ylabel="True Label",
            xlabel="Predicted Label",
            title=title,
        )
        plt.setp(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")

        threshold = matrix.max() / 2.0 if matrix.size and matrix.max() > 0 else 0.5
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax.text(
                    j,
                    i,
                    format(value, fmt),
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=6,
                )

        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
        plt.close(fig)

# ==============================================================================
# 1. 核心架构：动态残差路由 (必须与训练脚本完全一致，保证加载不报错)
# ==============================================================================
class ChannelGating(nn.Module):
    def __init__(self, dim, num_anchors):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, max(32, dim // 4)),
            nn.GELU(),
            nn.Linear(max(32, dim // 4), num_anchors * dim),
        )
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)
        self.num_anchors = num_anchors

    def forward(self, x):
        B, C = x.shape[0], x.shape[-1]
        x_pooled = x.reshape(B, -1, C).mean(dim=1)
        gates = torch.softmax(self.fc(x_pooled).reshape(B, self.num_anchors, C), dim=1)
        broadcast_shape = [B, self.num_anchors] + [1] * (x.dim() - 2) + [C]
        return gates.reshape(*broadcast_shape)

class DynamicResidualStageWrapper(nn.Module):
    def __init__(self, orig_stage, anchor_indices=(1, 4, 9), target_indices=(11, 14, 17), dim=512):
        super().__init__()
        self.orig_stage = orig_stage
        self.anchor_indices = anchor_indices
        self.target_indices = target_indices

        self.routers = nn.ModuleDict({
            str(idx): ChannelGating(dim, len(anchor_indices)) for idx in target_indices
        })
        self.gammas = nn.ParameterDict({
            str(idx): nn.Parameter(torch.tensor(float(INIT_DCA_GAMMA))) for idx in target_indices
        })

    def forward(self, x):
        # 严格遵守：先 downsample，再进 blocks
        if hasattr(self.orig_stage, 'downsample') and self.orig_stage.downsample is not None:
            x = self.orig_stage.downsample(x)

        anchors = []
        
        for i, block in enumerate(self.orig_stage.blocks):
            next_x = block(x)

            if i in self.target_indices:
                router = self.routers[str(i)]
                gamma = self.gammas[str(i)]
                gates = router(next_x)
                
                routed_feat = torch.zeros_like(next_x)
                for j, anchor_feat in enumerate(anchors):
                    routed_feat = routed_feat + gates[:, j] * anchor_feat
                
                next_x = next_x + gamma * routed_feat

            x = next_x

            if i in self.anchor_indices:
                anchors.append(x)

        return x

def inject_dynamic_residual_routing(model):
    orig_stage_3 = model.layers[2]
    wrapper = DynamicResidualStageWrapper(
        orig_stage=orig_stage_3,
        anchor_indices=(1, 4, 9),      
        target_indices=(11, 14, 17),   
        dim=512
    )
    model.layers[2] = wrapper
    return model

def normalize_dca_checkpoint_for_model(checkpoint, model):
    """兼容旧版 DCA 权重命名：fc1/fc2、dca_gammas -> 当前 fc.0/fc.2、gammas。"""
    normalized = {}
    for key, value in checkpoint.items():
        new_key = key
        if ".routers." in new_key:
            new_key = new_key.replace(".fc1.", ".fc.0.")
            new_key = new_key.replace(".fc2.", ".fc.2.")
        new_key = new_key.replace(".dca_gammas.", ".gammas.")
        normalized[new_key] = value

    model_state = model.state_dict()
    for key, value in model_state.items():
        if key.startswith("layers.2.gammas.") and key not in normalized:
            normalized[key] = value

    return normalized

# ==============================================================================
# 2. 评估主循环
# ==============================================================================
def main():
    if not os.path.exists(DATA_DIR):
        print(f"❌ 错误: 数据集路径不存在 {DATA_DIR}")
        return
    if not os.path.exists(WEIGHT_PATH):
        print(f"❌ 错误: 权重文件不存在 {WEIGHT_PATH}")
        return

    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=eval_transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    class_names = dataset.classes
    
    print("\n" + "="*80)
    print(f"🏁 开始评估消融实验模型:【Swin + DCA 动态残差路由】")
    print(f"✅ 数据集类别数: {len(class_names)} | 纯净测试集样本总数: {len(dataset)}")
    print(f"📥 加载权重: {WEIGHT_PATH}")
    print(f"📁 结果输出目录: {OUTPUT_DIR}")

    # 1. 加载原生模型
    model = timm.create_model(MODEL_ARCH, pretrained=False, num_classes=len(class_names))
    
    # 2. 注入动态残差路由 (不碰注意力)
    model = inject_dynamic_residual_routing(model)
    
    # 3. 加载权重
    try:
        checkpoint = torch.load(WEIGHT_PATH, map_location=DEVICE)
        if list(checkpoint.keys())[0].startswith('module.'):
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        checkpoint = normalize_dca_checkpoint_for_model(checkpoint, model)
        model.load_state_dict(checkpoint, strict=True)
        print("✅ 权重完美融合！")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return

    model.to(DEVICE)
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction='none')

    all_preds = []
    all_labels = []
    class_loss_sum = {i: 0.0 for i in range(len(class_names))}
    class_count = {i: 0 for i in range(len(class_names))}

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="正在全量扫描测试集"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            losses = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(inputs.size(0)):
                label_idx = labels[i].item()
                loss_val = losses[i].item()
                class_loss_sum[label_idx] += loss_val
                class_count[label_idx] += 1

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(len(class_names)), zero_division=0
    )
    
    report_lines = []
    report_lines.append("=== 大豆品种分类：Swin + DCA 评估报告 ===")
    report_lines.append(f"测试集路径: {DATA_DIR}")
    report_lines.append(f"权重路径: {WEIGHT_PATH}")
    report_lines.append(f"类别数: {len(class_names)}")
    report_lines.append(f"样本数: {len(dataset)}")
    report_lines.append("\n" + "-"*92)
    header = f"{'类别 (Class)':<11} | {'样本数':<5} | {'精确率 (P)':<9} | {'召回率 (R)':<9} | {'F1分数':<8} | {'平均 Loss'}"
    report_lines.append(header)
    report_lines.append("-" * 92)

    results = []
    for i, name in enumerate(class_names):
        avg_loss = class_loss_sum[i] / class_count[i] if class_count[i] > 0 else 0.0
        results.append({
            "name": name,
            "support": support[i],
            "p": precision[i],
            "r": recall[i],
            "f1": f1[i],
            "loss": avg_loss
        })

    results.sort(key=lambda x: x['f1'], reverse=False)

    for row in results:
        report_lines.append(f"{row['name']:<12} | {row['support']:<6} | {row['p']*100:>6.2f}%   | {row['r']*100:>6.2f}%   | {row['f1']*100:>6.2f}%   | {row['loss']:.4f}")

    avg_p = np.average(precision, weights=support)
    avg_r = np.average(recall, weights=support)
    avg_f1 = np.average(f1, weights=support)
    
    total_loss = sum(class_loss_sum.values())
    total_count = sum(class_count.values())
    global_avg_loss = total_loss / total_count if total_count > 0 else 0

    report_lines.append("-" * 92)
    report_lines.append(f"{'加权平均/总计':<10} | {total_count:<6} | {avg_p*100:>6.2f}%   | {avg_r*100:>6.2f}%   | {avg_f1*100:>6.2f}%   | {global_avg_loss:.4f}")
    report_lines.append("================================================================================")
    output_text = "\n".join(report_lines)
    print(output_text + "\n")

    summary = {
        "model": "swin_dca_only",
        "data_dir": DATA_DIR,
        "weight_path": WEIGHT_PATH,
        "num_classes": len(class_names),
        "num_samples": len(dataset),
        "weighted_precision": float(avg_p),
        "weighted_recall": float(avg_r),
        "weighted_f1": float(avg_f1),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "avg_loss": float(global_avg_loss),
    }
    save_evaluation_results(report_lines, results, summary, all_labels, all_preds, dataset)
    save_confusion_matrices(all_labels, all_preds, class_names)

if __name__ == "__main__":
    main()
