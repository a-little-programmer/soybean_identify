# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VIT_TRAIN_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../train/vit"))
if VIT_TRAIN_DIR not in sys.path:
    sys.path.insert(0, VIT_TRAIN_DIR)

from evaluate_report_utils import (  # noqa: E402
    format_eval_report,
    save_multiple_confusion_matrix_counts,
    write_report,
)
from vit_diff_dca_model import get_vit_diff_dca_model  # noqa: E402


# ================= 配置区域：评估变换与 ViT baseline 对齐 =================
BASE_DIR = CURRENT_DIR
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/classifier_dataset_hsv/test"))
CHECKPOINT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../result/checkpoints"))
WEIGHT_PATH = os.path.join(CHECKPOINT_DIR, "best_vit_diff_dca_soybean.pth")
CLASS_INDEX_PATH = os.path.join(CHECKPOINT_DIR, "class_indices_vit_diff_dca.json")

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EVAL_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(BASE_DIR, EVAL_NAME)
REPORT_FILE = os.path.join(OUTPUT_DIR, "report.txt")
# ========================================================================


def load_model(num_classes):
    print("构建 ViT-Diff-DCA 架构以加载权重...")
    model = get_vit_diff_dca_model(num_classes, pretrained=False)
    checkpoint = torch.load(WEIGHT_PATH, map_location=DEVICE)
    if list(checkpoint.keys())[0].startswith("module."):
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=True)
    model.to(DEVICE)
    model.eval()
    print(f"权重加载成功: {WEIGHT_PATH}")
    return model


def evaluate():
    eval_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if not os.path.exists(DATA_DIR):
        print(f"错误: 数据集路径不存在 {DATA_DIR}")
        return None
    if not os.path.exists(WEIGHT_PATH):
        print(f"错误: 权重路径不存在 {WEIGHT_PATH}")
        return None

    dataset = datasets.ImageFolder(DATA_DIR, transform=eval_transforms)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    model = load_model(len(dataset.classes))

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_preds, all_labels = [], []
    class_loss_sum = {i: 0.0 for i in range(len(dataset.classes))}
    class_count = {i: 0 for i in range(len(dataset.classes))}

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating vit_diff_dca"):
            inputs = inputs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
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
        all_labels,
        all_preds,
        labels=range(len(dataset.classes)),
        zero_division=0,
    )

    rows = []
    for i, name in enumerate(dataset.classes):
        avg_loss = class_loss_sum[i] / class_count[i] if class_count[i] > 0 else 0.0
        rows.append({
            "name": name,
            "support": support[i],
            "p": precision[i],
            "r": recall[i],
            "f1": f1[i],
            "loss": avg_loss,
        })
    rows.sort(key=lambda x: x["f1"], reverse=False)

    weighted_p = float(np.average(precision, weights=support))
    weighted_r = float(np.average(recall, weights=support))
    weighted_f1 = float(np.average(f1, weights=support))
    total_loss = sum(class_loss_sum.values())
    total_count = sum(class_count.values())
    avg_loss = total_loss / total_count if total_count > 0 else 0.0
    accuracy = float(np.mean(np.array(all_labels) == np.array(all_preds))) if all_labels else 0.0

    report_lines = format_eval_report(
        model_name="vit_diff_dca",
        data_dir=DATA_DIR,
        weight_path=WEIGHT_PATH,
        class_index_path=CLASS_INDEX_PATH,
        num_classes=len(dataset.classes),
        num_samples=len(dataset),
        accuracy=accuracy,
        macro_precision=float(np.mean(precision)),
        macro_recall=float(np.mean(recall)),
        macro_f1=float(np.mean(f1)),
        weighted_precision=weighted_p,
        weighted_recall=weighted_r,
        weighted_f1=weighted_f1,
        avg_loss=float(avg_loss),
        rows=rows,
    )

    print("\n".join(report_lines))
    write_report(report_lines, OUTPUT_DIR, append=False)

    return {
        "model": "vit_diff_dca",
        "labels": all_labels,
        "preds": all_preds,
        "class_names": dataset.classes,
    }


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    item = evaluate()
    if item is not None:
        save_multiple_confusion_matrix_counts([item], OUTPUT_DIR)
    print(f"\n评估输出目录: {OUTPUT_DIR}")
