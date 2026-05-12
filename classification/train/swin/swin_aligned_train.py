# -*- coding: utf-8 -*-
import os
import gc
import json
import time
import math
import random
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.metrics import f1_score

import timm

warnings.filterwarnings("ignore")

# ==============================
# 0. 基础配置 (与 swin_diff_train.py 对齐)
# ==============================
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../data/classifier_dataset_hsv"))
SAVE_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../result/checkpoints"))
MODEL_NAME = "best_swin_aligned.pth"
CLASS_INDEX_NAME = "class_indices_swin_aligned.json"

MODEL_ID = "swin_base_patch4_window7_224.ms_in22k_ft_in1k"
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 30
NUM_WORKERS = 4

LR_BACKBONE = 1e-5
LR_HEAD = 3e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.05
WARMUP_EPOCHS = 5
FREEZE_EPOCHS = 3

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==============================================================================
# 1. 数据加载
# ==============================================================================
def build_dataloaders():
    tf_train = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tf_val = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), tf_train)
    val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), tf_val)
    loaders = {
        'train': DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        ),
        'val': DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        ),
    }
    return train_ds, loaders

# ==============================================================================
# 2. 模型与优化器
# ==============================================================================
def get_model(num_classes):
    print("构建纯 Swin aligned baseline...")
    return timm.create_model(MODEL_ID, pretrained=True, num_classes=num_classes)

def get_fast_keywords():
    return ['head.']

def build_optimizer(model):
    fast_keys = get_fast_keywords()
    p_backbone = [p for n, p in model.named_parameters() if not any(k in n for k in fast_keys)]
    p_head = [p for n, p in model.named_parameters() if any(k in n for k in fast_keys)]
    return optim.AdamW(
        [{'params': p_backbone, 'lr': LR_BACKBONE}, {'params': p_head, 'lr': LR_HEAD}],
        weight_decay=WEIGHT_DECAY,
    )

def build_scheduler(optimizer):
    def lr_bb(epoch):
        if epoch < FREEZE_EPOCHS:
            return 0.0
        adj = epoch - FREEZE_EPOCHS
        if adj < WARMUP_EPOCHS:
            return max(0.05, adj / max(1, WARMUP_EPOCHS))
        prog = (adj - WARMUP_EPOCHS) / max(1, NUM_EPOCHS - FREEZE_EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    def lr_head(epoch):
        if epoch < WARMUP_EPOCHS:
            return max(0.05, epoch / max(1, WARMUP_EPOCHS))
        prog = (epoch - WARMUP_EPOCHS) / max(1, NUM_EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    return optim.lr_scheduler.LambdaLR(optimizer, [lr_bb, lr_head])

def build_criterion(train_ds, num_classes):
    counts = np.bincount(train_ds.targets, minlength=num_classes)
    class_weights = 1.0 / np.sqrt(counts + 1e-6)
    class_weights = class_weights / class_weights.mean()
    return nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE),
        label_smoothing=LABEL_SMOOTHING,
    )

# ==============================================================================
# 3. 训练主流程
# ==============================================================================
def main():
    set_seed(SEED)
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    os.makedirs(SAVE_DIR, exist_ok=True)

    train_ds, loaders = build_dataloaders()
    num_classes = len(train_ds.classes)

    model = get_model(num_classes).to(DEVICE)
    criterion = build_criterion(train_ds, num_classes)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    fast_keys = get_fast_keywords()
    best_f1 = -1.0

    print(f"Device: {DEVICE} | AMP: {USE_AMP}")
    print(f"Classes: {num_classes} | Train: {len(train_ds)} | Val: {len(loaders['val'].dataset)}")

    for epoch in range(NUM_EPOCHS):
        is_freeze = epoch < FREEZE_EPOCHS
        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")

        if is_freeze and epoch == 0:
            print("开启物理冻结，仅训练分类头...")
            for name, param in model.named_parameters():
                if not any(k in name for k in fast_keys):
                    param.requires_grad = False
        elif epoch == FREEZE_EPOCHS:
            print("解冻 Backbone，开始联合微调...")
            for param in model.parameters():
                param.requires_grad = True

        for phase in ['train', 'val']:
            is_train = phase == 'train'
            if is_train:
                if is_freeze:
                    model.eval()
                    for name, module in model.named_modules():
                        if any(k in name for k in fast_keys):
                            module.train()
                else:
                    model.train()
            else:
                model.eval()

            running_loss = 0.0
            preds_all, labels_all = [], []
            pbar = tqdm(loaders[phase], desc=phase.capitalize(), leave=False)

            for inputs, labels in pbar:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                if is_train:
                    optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(is_train):
                    with torch.amp.autocast('cuda', enabled=USE_AMP):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    if is_train:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        active_params = [p for p in model.parameters() if p.requires_grad]
                        torch.nn.utils.clip_grad_norm_(active_params, max_norm=5.0)
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                preds_all.extend(torch.argmax(outputs, 1).detach().cpu().numpy())
                labels_all.extend(labels.detach().cpu().numpy())
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            if is_train:
                scheduler.step()

            epoch_loss = running_loss / len(loaders[phase].dataset)
            epoch_acc = 100.0 * (np.array(preds_all) == np.array(labels_all)).mean()
            epoch_f1 = 100.0 * f1_score(
                labels_all,
                preds_all,
                labels=list(range(num_classes)),
                average='macro',
                zero_division=0,
            )

            print(
                f"{phase.upper()} | Loss: {epoch_loss:.4f} | "
                f"Acc: {epoch_acc:.2f}% | F1: {epoch_f1:.2f}%"
            )

            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, MODEL_NAME))
                print(f"保存当前最佳模型: {MODEL_NAME}")

    print(f"训练结束，最佳 Val F1: {best_f1:.2f}%")
    with open(os.path.join(SAVE_DIR, CLASS_INDEX_NAME), 'w', encoding='utf-8') as f:
        json.dump(train_ds.class_to_idx, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
