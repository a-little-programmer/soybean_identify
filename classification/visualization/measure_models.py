# -*- coding: utf-8 -*-
import csv
import os
import sys
import time

import torch
import torch.nn as nn
from torchvision import models
import timm

EVALUATE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../evaluate"))
if EVALUATE_DIR not in sys.path:
    sys.path.insert(0, EVALUATE_DIR)

from evaluate_swin_diff import inject_only_diff
from evaluate_swin_dca import inject_dynamic_residual_routing
from evaluate_swin_diff_dca import get_model as get_swin_diff_dca


# ==============================================================================
# 配置区域
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "model_efficiency")
REPORT_FILE = os.path.join(OUTPUT_DIR, "model_efficiency_report.txt")
CSV_FILE = os.path.join(OUTPUT_DIR, "model_efficiency_metrics.csv")

NUM_CLASSES = 25
IMAGE_SIZE = 224
BATCH_SIZE = 1

# "cpu" 表示统一测 CPU；"cuda" 表示有 GPU 时测 GPU；"auto" 表示优先 GPU。
DEVICE_MODE = "cpu"

WARMUP_ITERS = 20
MEASURE_ITERS = 100

MODEL_ID = "swin_base_patch4_window7_224.ms_in22k_ft_in1k"


def get_device():
    if DEVICE_MODE == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if DEVICE_MODE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def build_resnet50():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def build_regnet():
    model = models.regnet_y_3_2gf(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def build_vit():
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    return model


def build_swin_baseline():
    return timm.create_model(MODEL_ID, pretrained=False, num_classes=NUM_CLASSES)


def build_swin_diff():
    model = timm.create_model(MODEL_ID, pretrained=False, num_classes=NUM_CLASSES)
    return inject_only_diff(model)


def build_swin_dca():
    model = timm.create_model(MODEL_ID, pretrained=False, num_classes=NUM_CLASSES)
    return inject_dynamic_residual_routing(model)


def build_models():
    return [
        ("ResNet-50", build_resnet50),
        ("RegNetY-3.2GF", build_regnet),
        ("ViT-B/16", build_vit),
        ("Swin-Base", build_swin_baseline),
        ("Swin + Diff", build_swin_diff),
        ("Swin + DCA", build_swin_dca),
        ("Swin + Diff + DCA", lambda: get_swin_diff_dca(NUM_CLASSES)),
    ]


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 ** 2)
    return total_params, trainable_params, model_size_mb


def synchronize_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_inference_time(model, device):
    dummy_input = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    model.eval()

    with torch.no_grad():
        for _ in range(WARMUP_ITERS):
            _ = model(dummy_input)
        synchronize_if_needed(device)

        start_time = time.perf_counter()
        for _ in range(MEASURE_ITERS):
            _ = model(dummy_input)
        synchronize_if_needed(device)
        end_time = time.perf_counter()

    avg_time_ms = (end_time - start_time) * 1000.0 / MEASURE_ITERS
    return avg_time_ms


def measure_model(model_name, build_fn, device):
    print(f"正在统计: {model_name}")
    model = build_fn().to(device)
    total_params, trainable_params, model_size_mb = count_parameters(model)
    avg_time_ms = measure_inference_time(model, device)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "params_m": total_params / 1e6,
        "trainable_params_m": trainable_params / 1e6,
        "model_size_mb": model_size_mb,
        "inference_time_ms": avg_time_ms,
        "device": str(device),
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
    }


def format_report(rows):
    lines = [
        "=== 大豆品种分类模型参数量与单张推理时间统计 ===",
        f"Device: {rows[0]['device'] if rows else '-'}",
        f"Input: batch={BATCH_SIZE}, shape=3x{IMAGE_SIZE}x{IMAGE_SIZE}",
        f"Warmup iters: {WARMUP_ITERS}",
        f"Measure iters: {MEASURE_ITERS}",
        "",
        "-" * 104,
        f"{'Model':<20} | {'Params (M)':>10} | {'Trainable (M)':>13} | {'Size (MB)':>9} | {'Infer Time (ms)':>15}",
        "-" * 104,
    ]

    for row in rows:
        lines.append(
            f"{row['model']:<20} | "
            f"{row['params_m']:>10.2f} | "
            f"{row['trainable_params_m']:>13.2f} | "
            f"{row['model_size_mb']:>9.2f} | "
            f"{row['inference_time_ms']:>15.2f}"
        )

    lines.append("-" * 104)
    lines.append("说明: 该脚本不加载训练权重、不读取数据集，只用随机 tensor 测模型结构的单张前向耗时。")
    return lines


def save_csv(rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fieldnames = [
        "model",
        "params_m",
        "trainable_params_m",
        "model_size_mb",
        "inference_time_ms",
        "device",
        "batch_size",
        "image_size",
    ]
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_report(lines):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    device = get_device()
    rows = []

    for model_name, build_fn in build_models():
        rows.append(measure_model(model_name, build_fn, device))

    report_lines = format_report(rows)
    print("\n".join(report_lines))

    save_report(report_lines)
    save_csv(rows)
    print(f"\n统计结果已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
