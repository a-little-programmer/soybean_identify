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
import warnings
import timm
from typing import Optional, Tuple
from timm.layers import to_2tuple, trunc_normal_

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. 配置区域 (绝对路径对齐)
# ==============================================================================
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/classifier_dataset_hsv/test")) 
WEIGHT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_swin_diff.pth")
EVAL_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(BASE_DIR, EVAL_NAME)

MODEL_ID = "swin_base_patch4_window7_224.ms_in22k_ft_in1k"
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_STAGE_DIMS = (1024,)
LITE_RATIO = 0.25
INIT_LAMBDA = -2.0           
INIT_DIFF_GAMMA = 0.02

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
# 1. 核心架构逻辑 (必须与训练代码100%一致，用于支撑结构注入)
# ==============================================================================
class DifferentialWindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, qkv_bias=True, attn_drop=0., proj_drop=0., device=None, dtype=None):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.dim, self.num_heads = dim, num_heads
        self.window_size = to_2tuple(window_size)
        self.window_area = self.window_size[0] * self.window_size[1]
        self.scale = (dim // num_heads) ** -0.5
        
        self.relative_position_bias_table = nn.Parameter(torch.empty((2*self.window_size[0]-1)*(2*self.window_size[1]-1), num_heads, **dd))
        self.register_buffer("relative_position_index", torch.empty(self.window_area, self.window_area, device=device, dtype=torch.long), persistent=False)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, **dd)
        self.attn_drop, self.proj = nn.Dropout(attn_drop), nn.Linear(dim, dim, **dd)
        self.proj_drop, self.softmax = nn.Dropout(proj_drop), nn.Softmax(dim=-1)

        self.use_diff = dim in TARGET_STAGE_DIMS
        if self.use_diff:
            head_dim_lite = max(1, int((dim // num_heads) * LITE_RATIO))
            self.lite_dim = head_dim_lite * num_heads
            self.q_lite = nn.Linear(dim, self.lite_dim, bias=False, **dd)
            self.k_lite = nn.Linear(dim, self.lite_dim, bias=False, **dd)
            self.v_lite = nn.Linear(dim, self.lite_dim, bias=False, **dd)
            self.diff_proj = nn.Linear(self.lite_dim, dim, **dd)
            self.diff_gamma = nn.Parameter(torch.tensor(float(INIT_DIFF_GAMMA), **dd))
            self.lambda_1_raw = nn.Parameter(torch.tensor(float(INIT_LAMBDA), **dd))
            self.lambda_2_raw = nn.Parameter(torch.tensor(float(INIT_LAMBDA), **dd))
            trunc_normal_(self.diff_proj.weight, std=1e-4)
            nn.init.zeros_(self.diff_proj.bias)

    def _get_rel_pos_bias(self):
        return self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1).permute(2, 0, 1).contiguous().unsqueeze(0)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        rel_pos_bias = self._get_rel_pos_bias()
        attn = (q * self.scale) @ k.transpose(-2, -1) + rel_pos_bias
        
        if mask is not None: 
            attn = attn.view(-1, mask.shape[0], self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn_soft = self.softmax(attn)
        x_out = (self.attn_drop(attn_soft) @ v).transpose(1, 2).reshape(B, N, C)
        
        if self.use_diff:
            head_dim_lite = self.lite_dim // self.num_heads
            q_l = self.q_lite(x).reshape(B, N, self.num_heads, head_dim_lite).permute(0, 2, 1, 3)
            k_l = self.k_lite(x).reshape(B, N, self.num_heads, head_dim_lite).permute(0, 2, 1, 3)
            v_l = self.v_lite(x).reshape(B, N, self.num_heads, head_dim_lite).permute(0, 2, 1, 3)
            
            attn_l = (q_l * (head_dim_lite**-0.5)) @ k_l.transpose(-2, -1) + rel_pos_bias
            
            lam1 = torch.sigmoid(self.lambda_1_raw)
            lam2 = torch.sigmoid(self.lambda_2_raw)
            diff = lam1 * self.softmax(attn_l) - lam2 * attn_soft.detach()
            
            delta_v = (self.attn_drop(diff) @ v_l).transpose(1, 2).reshape(B, N, -1)
            x_out = x_out + self.diff_gamma * self.diff_proj(delta_v)
            
        return self.proj_drop(self.proj(x_out))

def inject_only_diff(model):
    print("🔧 评估准备：原位注入 Diff Attention 结构...")
    layer = model.layers[3]
    for block in layer.blocks:
        old = block.attn
        new = DifferentialWindowAttention(old.qkv.in_features, old.num_heads, old.window_size, old.qkv.bias is not None)
        new.load_state_dict(old.state_dict(), strict=False)
        new.relative_position_index.copy_(old.relative_position_index)
        block.attn = new.to(old.qkv.weight.device)
    return model

# ==============================================================================
# 2. 评估主程序
# ==============================================================================
def main():
    if not os.path.exists(DATA_DIR):
        print(f"❌ 测试集路径不存在: {DATA_DIR}")
        return
    if not os.path.exists(WEIGHT_PATH):
        print(f"❌ 权重文件不存在: {WEIGHT_PATH}")
        return

    # 测试集严谨预处理 (与训练集绝对对齐尺寸和均值方差)
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=test_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    class_names = dataset.classes
    num_classes = len(class_names)
    
    print("\n" + "="*85)
    print(f"🏁 开始评估消融组:【仅 Diff Attention 版】")
    print(f"✅ 数据集类别数: {num_classes} | 待测试样本数: {len(dataset)}")
    print(f"📁 结果输出目录: {OUTPUT_DIR}")

    # 1. 构建基础模型 (pretrained=False 即可，因为我们要加载权重)
    model = timm.create_model(MODEL_ID, pretrained=False, num_classes=num_classes)
    
    # 2. 注入魔改结构 (⚠️ 极度重要：必须在 load_state_dict 之前)
    model = inject_only_diff(model)
    
    # 3. 加载训练好的权重
    try:
        checkpoint = torch.load(WEIGHT_PATH, map_location=DEVICE)
        # 兼容潜在的 DataParallel (module. prefix)
        if list(checkpoint.keys())[0].startswith('module.'):
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        # strict=True 确保我们注入的结构和保存的权重严丝合缝！
        model.load_state_dict(checkpoint, strict=True)
        print(f"✅ 权重完美匹配并加载: {WEIGHT_PATH}")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        return

    model.to(DEVICE)
    model.eval()
    
    # reduction='none' 方便我们后面统计每个类的平均 Loss
    criterion = nn.CrossEntropyLoss(reduction='none')

    all_preds, all_labels = [], []
    class_loss_sum = {i: 0.0 for i in range(num_classes)}
    class_count = {i: 0 for i in range(num_classes)}

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="正在全量扫描测试集"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            with torch.amp.autocast('cuda', enabled=DEVICE.type == "cuda"):
                outputs = model(images)
                losses = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(images.size(0)):
                label_idx = labels[i].item()
                class_loss_sum[label_idx] += losses[i].item()
                class_count[label_idx] += 1

    # 计算各项指标 (zero_division=0 防止除以0警告)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(num_classes)), zero_division=0
    )
    
    report_lines = []
    report_lines.append("=== 大豆品种分类：Swin + Diff 评估报告 ===")
    report_lines.append(f"测试集路径: {DATA_DIR}")
    report_lines.append(f"权重路径: {WEIGHT_PATH}")
    report_lines.append(f"类别数: {num_classes}")
    report_lines.append(f"样本数: {len(dataset)}")
    report_lines.append("\n" + "-"*92)
    report_lines.append(f"{'类别 (Class)':<15} | {'样本数':<5} | {'精确率 (P)':<9} | {'召回率 (R)':<9} | {'F1分数':<8} | {'平均 Loss'}")
    report_lines.append("-" * 92)

    results = []
    for i, name in enumerate(class_names):
        avg_loss = class_loss_sum[i] / class_count[i] if class_count[i] > 0 else 0.0
        results.append({"name": name, "support": support[i], "p": precision[i], "r": recall[i], "f1": f1[i], "loss": avg_loss})

    # 按 F1 分数从小到大排序输出，方便你直接一眼揪出“钉子户”大豆
    results.sort(key=lambda x: x['f1'], reverse=False)
    for row in results:
        report_lines.append(f"{row['name']:<16} | {row['support']:<6} | {row['p']*100:>6.2f}%   | {row['r']*100:>6.2f}%   | {row['f1']*100:>6.2f}%   | {row['loss']:.4f}")

    avg_p, avg_r, avg_f1 = np.average(precision, weights=support), np.average(recall, weights=support), np.average(f1, weights=support)
    macro_f1 = np.mean(f1)
    total_count = sum(class_count.values())
    global_avg_loss = sum(class_loss_sum.values()) / total_count if total_count > 0 else 0

    report_lines.append("-" * 92)
    report_lines.append(f"{'加权平均 (Weighted)':<18} | {total_count:<6} | {avg_p*100:>6.2f}%   | {avg_r*100:>6.2f}%   | {avg_f1*100:>6.2f}%   | {global_avg_loss:.4f}")
    report_lines.append(f"{'宏平均 (Macro)':<18} | {'-':<6} | {np.mean(precision)*100:>6.2f}%   | {np.mean(recall)*100:>6.2f}%   | {macro_f1*100:>6.2f}%   | {'-':<6}")
    report_lines.append("=================================================================================")
    output_text = "\n".join(report_lines)
    print(output_text + "\n")

    summary = {
        "model": "swin_diff",
        "data_dir": DATA_DIR,
        "weight_path": WEIGHT_PATH,
        "num_classes": num_classes,
        "num_samples": len(dataset),
        "weighted_precision": float(avg_p),
        "weighted_recall": float(avg_r),
        "weighted_f1": float(avg_f1),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(macro_f1),
        "avg_loss": float(global_avg_loss),
    }
    save_evaluation_results(report_lines, results, summary, all_labels, all_preds, dataset)
    save_confusion_matrices(all_labels, all_preds, class_names)

if __name__ == "__main__":
    main()
