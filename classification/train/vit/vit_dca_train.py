# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import time
import copy
import json
import gc
import random
import numpy as np
from tqdm import tqdm
import math
from sklearn.metrics import f1_score

from vit_diff_dca_model import get_fast_keywords, get_vit_dca_model

# ================= 配置区域：与 classification/train/vit/vit_train.py 对齐 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../data/classifier_dataset_hsv"))
SAVE_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../result/checkpoints"))
MODEL_NAME = "best_vit_dca_soybean.pth"
CLASS_INDEX_NAME = "class_indices_vit_dca.json"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 30
NUM_WORKERS = 4
LR_BACKBONE = 1e-5
LR_NEW = 3e-4
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.05
WARMUP_EPOCHS = 5
FREEZE_EPOCHS = 3
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
# ======================================================================


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_scheduler(optimizer):
    def lr_bb(epoch):
        if epoch < FREEZE_EPOCHS:
            return 0.0
        adj = epoch - FREEZE_EPOCHS
        if adj < WARMUP_EPOCHS:
            return max(0.05, adj / max(1, WARMUP_EPOCHS))
        progress = float(adj - WARMUP_EPOCHS) / float(max(1, NUM_EPOCHS - FREEZE_EPOCHS - WARMUP_EPOCHS))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def lr_new(epoch):
        if epoch < WARMUP_EPOCHS:
            return max(0.05, epoch / max(1, WARMUP_EPOCHS))
        progress = float(epoch - WARMUP_EPOCHS) / float(max(1, NUM_EPOCHS - WARMUP_EPOCHS))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, [lr_bb, lr_new])


def compute_class_weights(dataset, num_classes):
    counts = np.bincount(dataset.targets, minlength=num_classes).astype(np.float32)
    weights = 1.0 / np.sqrt(counts + 1e-6)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def main():
    set_seed(SEED)
    gc.collect()
    torch.cuda.empty_cache()
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"设备: {DEVICE} | 启用 AMP 混合精度: {USE_AMP}")
    print(f"数据集: {DATA_DIR}")
    print("实验: ViT-B/16 + DCA，训练参数与 vit_train.py 对齐")

    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("错误: 找不到 train 或 val 目录。")
        return

    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
        "val": datasets.ImageFolder(val_dir, transform=data_transforms["val"]),
    }

    num_classes = len(image_datasets["train"].classes)
    print(
        f"检测到 {num_classes} 个类别 | "
        f"Train: {len(image_datasets['train'])} | Val: {len(image_datasets['val'])}"
    )

    dataloaders = {
        "train": DataLoader(
            image_datasets["train"],
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        ),
        "val": DataLoader(
            image_datasets["val"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        ),
    }

    model = get_vit_dca_model(num_classes, pretrained=True).to(DEVICE)

    class_weights = compute_class_weights(image_datasets["train"], num_classes)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    fast_keys = get_fast_keywords()
    p_backbone = [p for n, p in model.named_parameters() if not any(k in n for k in fast_keys)]
    p_new = [p for n, p in model.named_parameters() if any(k in n for k in fast_keys)]
    optimizer = optim.AdamW(
        [{"params": p_backbone, "lr": LR_BACKBONE}, {"params": p_new, "lr": LR_NEW}],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = build_scheduler(optimizer)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    since = time.time()
    best_f1 = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(NUM_EPOCHS):
        is_freeze = epoch < FREEZE_EPOCHS
        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")
        print("-" * 40)

        if is_freeze and epoch == 0:
            print("开启物理冻结 (仅训练分类头与 DCA 路由)...")
            for n, p in model.named_parameters():
                if not any(k in n for k in fast_keys):
                    p.requires_grad = False
        elif epoch == FREEZE_EPOCHS:
            print("解冻 Backbone...")
            for p in model.parameters():
                p.requires_grad = True

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Current LR: {current_lr:.7f}")

        for phase in ["train", "val"]:
            if phase == "train":
                if is_freeze:
                    model.eval()
                    for module_name, module in model.named_modules():
                        if any(k.rstrip(".") in module_name for k in fast_keys):
                            module.train()
                    model.heads.head.train()
                else:
                    model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            preds_all, labels_all = [], []
            pbar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()}", leave=False)

            for inputs, labels in pbar:
                inputs = inputs.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    with torch.amp.autocast("cuda", enabled=USE_AMP):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        active_params = [p for p in model.parameters() if p.requires_grad]
                        torch.nn.utils.clip_grad_norm_(active_params, max_norm=5.0)
                        scaler.step(optimizer)
                        scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                preds_all.extend(preds.detach().cpu().numpy().tolist())
                labels_all.extend(labels.detach().cpu().numpy().tolist())
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            epoch_f1 = 100.0 * f1_score(
                labels_all,
                preds_all,
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            )

            print(
                f"{phase.capitalize()} Loss: {epoch_loss:.4f} | "
                f"Acc: {epoch_acc:.4f} | Macro-F1: {epoch_f1:.2f}%"
            )

            if phase == "val" and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                save_path = os.path.join(SAVE_DIR, MODEL_NAME)
                torch.save(best_model_wts, save_path)
                print(f"保存当前最佳 ViT-DCA 模型: {save_path}")

        scheduler.step()

    time_elapsed = time.time() - since
    print(f"\n训练完成，耗时: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"最佳验证 Macro-F1: {best_f1:.2f}%")

    idx_path = os.path.join(SAVE_DIR, CLASS_INDEX_NAME)
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(image_datasets["train"].class_to_idx, f, ensure_ascii=False, indent=2)
    print(f"类别索引映射已保存至: {idx_path}")


if __name__ == "__main__":
    main()
