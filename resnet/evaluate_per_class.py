# -*- coding: utf-8 -*-
import os
import csv
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import timm  # 🌟 核心修复：必须导入 timm

# ==============================================================================
# 配置区域
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/classifier_dataset_hsv/test"))  # 请确保指向你的测试集

MODEL_PATH = [
    os.path.join(BASE_DIR, "checkpoints", "best_regnet_soybean.pth"),
    os.path.join(BASE_DIR, "checkpoints", "best_resnet50_soybean.pth"),
    os.path.join(BASE_DIR, "checkpoints", "best_vit_soybean.pth"),
    os.path.join(BASE_DIR, "checkpoints", "best_swin_soybean.pth"),
]

MODEL_ARCH = [
    "regnet",
    "resnet50",
    "vit",
    "swin",
]

BATCH_SIZE = 64 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EVAL_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(BASE_DIR, EVAL_NAME)
REPORT_FILE = os.path.join(OUTPUT_DIR, "report.txt")

def save_model_result_files(
    model_arch,
    results,
    summary,
    all_labels,
    all_preds,
    dataset,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    safe_name = model_arch.replace("/", "_").replace(" ", "_")
    per_class_path = os.path.join(OUTPUT_DIR, f"{safe_name}_per_class_metrics.csv")
    summary_path = os.path.join(OUTPUT_DIR, f"{safe_name}_summary.json")
    predictions_path = os.path.join(OUTPUT_DIR, f"{safe_name}_predictions.csv")

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

# ==============================================================================
# 1. 纯净加载器 (混合 torchvision 与 timm)
# ==============================================================================
def load_model(arch, num_classes, model_path):
    print(f"🏗️ 正在构建基础架构: {arch} ...")
    
    if arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif arch == "regnet":
        model = models.regnet_y_3_2gf(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif arch == "vit":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        
    elif arch == "swin":
        # 🌟 致命修复：彻底抛弃 torchvision 的 swin_v2_b！
        # 使用和训练时 100% 对应的 timm 底座
        model = timm.create_model("swin_base_patch4_window7_224.ms_in22k_ft_in1k", pretrained=False, num_classes=num_classes)
        
    else:
        raise ValueError(f"❌ 未知模型架构: {arch}")

    # 加载权重
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        if list(checkpoint.keys())[0].startswith('module.'):
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            
        model.load_state_dict(checkpoint, strict=True)
        print(f"✅ 权重 {model_path} 加载成功！")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        raise e 

    model.to(DEVICE)
    model.eval()
    return model

# ==============================================================================
# 2. 单模型评估流程
# ==============================================================================
def evaluate_single_model(model_path, model_arch):
    report_lines = []

    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if not os.path.exists(DATA_DIR):
        print(f"❌ 错误: 数据集路径不存在 {DATA_DIR}")
        return
        
    dataset = datasets.ImageFolder(DATA_DIR, transform=eval_transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    class_names = dataset.classes
    
    report_lines.append("\n" + "="*80)
    report_lines.append(f"🏁 开始评估基线模型: {model_arch}")
    report_lines.append(f"📁 结果输出目录: {OUTPUT_DIR}")
    
    try:
        model = load_model(model_arch, len(class_names), model_path)
    except Exception as e:
        skip_msg = "⏭️ 跳过当前模型的评估。"
        print(skip_msg)
        report_lines.append(skip_msg)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(REPORT_FILE, "a", encoding="utf-8") as f:
            f.write("\n".join(report_lines) + "\n")
        return

    criterion = nn.CrossEntropyLoss(reduction='none')
    all_preds, all_labels = [], []
    class_loss_sum = {i: 0.0 for i in range(len(class_names))}
    class_count = {i: 0 for i in range(len(class_names))}

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"Evaluating {model_arch}"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            with torch.amp.autocast('cuda', enabled=DEVICE.type == "cuda"):
                outputs = model(inputs)
                losses = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(inputs.size(0)):
                label_idx = labels[i].item()
                class_loss_sum[label_idx] += losses[i].item()
                class_count[label_idx] += 1

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(len(class_names)), zero_division=0
    )
    
    report_lines.append("\n" + "-"*92)
    header = f"{'类别 (Class)':<11} | {'样本数':<5} | {'精确率 (P)':<9} | {'召回率 (R)':<9} | {'F1分数':<8} | {'平均 Loss'}"
    report_lines.append(header)
    report_lines.append("-" * 92)

    results = []
    for i, name in enumerate(class_names):
        avg_loss = class_loss_sum[i] / class_count[i] if class_count[i] > 0 else 0.0
        results.append({"name": name, "support": support[i], "p": precision[i], "r": recall[i], "f1": f1[i], "loss": avg_loss})

    results.sort(key=lambda x: x['f1'], reverse=False)

    for row in results:
        line = f"{row['name']:<12} | {row['support']:<6} | {row['p']*100:>6.2f}%   | {row['r']*100:>6.2f}%   | {row['f1']*100:>6.2f}%   | {row['loss']:.4f}"
        report_lines.append(line)

    report_lines.append("-" * 92)

    avg_p, avg_r, avg_f1 = np.average(precision, weights=support), np.average(recall, weights=support), np.average(f1, weights=support)
    total_loss, total_count = sum(class_loss_sum.values()), sum(class_count.values())
    global_avg_loss = total_loss / total_count if total_count > 0 else 0

    summary_line = f"{'加权平均/总计':<10} | {total_count:<6} | {avg_p*100:>6.2f}%   | {avg_r*100:>6.2f}%   | {avg_f1*100:>6.2f}%   | {global_avg_loss:.4f}"
    report_lines.append(summary_line)
    report_lines.append("================================================================================\n")

    output_text = "\n".join(report_lines)
    print(output_text)

    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(output_text + "\n")

    summary = {
        "model": model_arch,
        "data_dir": DATA_DIR,
        "weight_path": model_path,
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
    save_model_result_files(model_arch, results, summary, all_labels, all_preds, dataset)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("=== 大豆品种分类：基线模型 (Baseline) 评估报告 ===\n")
        f.write(f"测试集路径: {DATA_DIR}\n\n")

    for path, arch in zip(MODEL_PATH, MODEL_ARCH):
        evaluate_single_model(model_path=path, model_arch=arch)
    
    print(f"\n🎉 恭喜！所有基线模型的评估结果已成功保存至: {OUTPUT_DIR}")
