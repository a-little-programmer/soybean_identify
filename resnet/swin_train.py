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
# 0. 基础配置 (消融实验基线对齐)
# ==============================
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/classifier_dataset_hsv"))
SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")
MODEL_NAME = "best_swin_soybean.pth"
CLASS_INDEX_NAME = "class_indices_swin_baseline.json"

# 统一使用 TIMM 的 Swin_V1_224
MODEL_ID = "swin_base_patch4_window7_224.ms_in22k_ft_in1k"
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 30
NUM_WORKERS = 4

LR_BACKBONE = 1e-5
LR_NEW = 3e-4  # 基线版的 new-parts 仅包含分类 head
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.05
WARMUP_EPOCHS = 5
FREEZE_EPOCHS = 3

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==============================================================================
# 1. 训练管道 (无任何结构魔改)
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

def get_fast_keywords():
    return ['head.']

def build_scheduler(optimizer):
    def lr_bb(epoch):
        if epoch < FREEZE_EPOCHS: return 0.0
        adj = epoch - FREEZE_EPOCHS
        if adj < WARMUP_EPOCHS: return max(0.05, adj / max(1, WARMUP_EPOCHS))
        prog = (adj - WARMUP_EPOCHS) / max(1, NUM_EPOCHS - FREEZE_EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    def lr_new(epoch):
        if epoch < WARMUP_EPOCHS: return max(0.05, epoch / max(1, WARMUP_EPOCHS))
        prog = (epoch - WARMUP_EPOCHS) / max(1, NUM_EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    return optim.lr_scheduler.LambdaLR(optimizer, [lr_bb, lr_new])

def main():
    set_seed(SEED); os.makedirs(SAVE_DIR, exist_ok=True)
    train_ds, loaders = build_dataloaders()
    num_classes = len(train_ds.classes)

    print("加载 Swin baseline 模型...")
    model = timm.create_model(MODEL_ID, pretrained=True, num_classes=num_classes).to(DEVICE)

    counts = np.bincount(train_ds.targets, minlength=num_classes)
    class_weights = 1.0 / np.sqrt(counts + 1e-6)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE),
        label_smoothing=LABEL_SMOOTHING,
    )

    fast_keys = get_fast_keywords()
    p_backbone = [p for n, p in model.named_parameters() if not any(k in n for k in fast_keys)]
    p_new = [p for n, p in model.named_parameters() if any(k in n for k in fast_keys)]
    optimizer = optim.AdamW(
        [{'params': p_backbone, 'lr': LR_BACKBONE}, {'params': p_new, 'lr': LR_NEW}],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = build_scheduler(optimizer)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    best_f1 = -1.0
    for epoch in range(NUM_EPOCHS):
        is_freeze = epoch < FREEZE_EPOCHS
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")

        # Baseline 版同样采用物理冻结 3 轮预热策略。
        if is_freeze and epoch == 0:
            print("开启物理冻结 (仅预热分类头)...")
            for n, p in model.named_parameters():
                if not any(k in n for k in fast_keys): p.requires_grad = False
        elif epoch == FREEZE_EPOCHS:
            print("解冻 Backbone...")
            for p in model.parameters(): p.requires_grad = True

        for phase in ['train', 'val']:
            is_train = phase == 'train'
            if is_train:
                if is_freeze:
                    model.eval()
                    for n, m in model.named_modules():
                        if any(k in n for k in fast_keys): m.train()
                else:
                    model.train()
            else:
                model.eval()

            running_loss = 0.0
            preds_all, labels_all = [], []
            pbar = tqdm(loaders[phase], desc=phase.capitalize(), leave=False)

            for inputs, labels in pbar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
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
                preds_all.extend(torch.argmax(outputs, 1).cpu().numpy())
                labels_all.extend(labels.cpu().numpy())
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
