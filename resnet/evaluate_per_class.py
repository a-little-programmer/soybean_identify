# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import timm  # 🌟 核心修复：必须导入 timm
from evaluate_report_utils import (
    format_eval_report,
    save_multiple_confusion_matrix_counts,
    write_report,
)

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
    return {
        "model": model_arch,
        "labels": all_labels,
        "preds": all_preds,
        "class_names": dataset.classes,
    }

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
    
    try:
        model = load_model(model_arch, len(class_names), model_path)
    except Exception as e:
        skip_msg = f"模型: {model_arch}\n权重路径: {model_path}\n状态: 权重加载失败，跳过当前模型。"
        print(skip_msg)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(REPORT_FILE, "a", encoding="utf-8") as f:
            f.write(skip_msg + "\n")
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
    
    results = []
    for i, name in enumerate(class_names):
        avg_loss = class_loss_sum[i] / class_count[i] if class_count[i] > 0 else 0.0
        results.append({"name": name, "support": support[i], "p": precision[i], "r": recall[i], "f1": f1[i], "loss": avg_loss})

    results.sort(key=lambda x: x['f1'], reverse=False)

    avg_p, avg_r, avg_f1 = np.average(precision, weights=support), np.average(recall, weights=support), np.average(f1, weights=support)
    total_loss, total_count = sum(class_loss_sum.values()), sum(class_count.values())
    global_avg_loss = total_loss / total_count if total_count > 0 else 0
    accuracy = float(np.mean(np.array(all_labels) == np.array(all_preds))) if all_labels else 0.0
    report_lines = format_eval_report(
        model_name=model_arch,
        data_dir=DATA_DIR,
        weight_path=model_path,
        class_index_path=None,
        num_classes=len(class_names),
        num_samples=len(dataset),
        accuracy=accuracy,
        macro_precision=float(np.mean(precision)),
        macro_recall=float(np.mean(recall)),
        macro_f1=float(np.mean(f1)),
        weighted_precision=float(avg_p),
        weighted_recall=float(avg_r),
        weighted_f1=float(avg_f1),
        avg_loss=float(global_avg_loss),
        rows=results,
    )

    output_text = "\n".join(report_lines)
    print(output_text)

    write_report(report_lines, OUTPUT_DIR, append=True)

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
    return save_model_result_files(model_arch, results, summary, all_labels, all_preds, dataset)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("")

    confusion_items = []
    for path, arch in zip(MODEL_PATH, MODEL_ARCH):
        item = evaluate_single_model(model_path=path, model_arch=arch)
        if item is not None:
            confusion_items.append(item)

    save_multiple_confusion_matrix_counts(confusion_items, OUTPUT_DIR)
    
    print(f"\n🎉 恭喜！所有基线模型的评估结果已成功保存至: {OUTPUT_DIR}")
