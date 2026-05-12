#!/bin/bash

# 1. 遇到任何错误立即停止执行
set -e

# 2. 确保进入脚本所在的目录 (防止找不到文件的路径问题)
cd "$(dirname "$0")"

echo "========================================"
echo "任务开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# --- 第一步 ---
echo "[1/4] 正在执行 regnet_train.py ..."
python /nfs/spy/soybean_detect/code/classification/train/regnet_train.py
echo ">>> regnet_train.py 执行成功"
echo "----------------------------------------"

# --- 第二步 ---
echo "[2/4] 正在执行 resnet_train.py ..."
python /nfs/spy/soybean_detect/code/classification/train/resnet_train.py
echo ">>> resnet_train.py 执行成功"
echo "----------------------------------------"

# --- 第三步 ---
echo "[3/4] 正在执行 swin_train.py ..."
python /nfs/spy/soybean_detect/code/classification/train/swin/swin_train.py
echo ">>> swin_train.py 执行成功"

# --- 第四步 ---
echo "[4/4] 正在执行 vit_train.py ..."
python /nfs/spy/soybean_detect/code/classification/train/vit/vit_train.py
echo ">>> vit_train.py 执行成功"

echo "========================================"
echo "所有任务已完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
