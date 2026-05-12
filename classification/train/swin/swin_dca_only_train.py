# -*- coding: utf-8 -*-
import os
import gc
import json
import time
import math
import copy
import random
import warnings
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import timm
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

# ==============================
# 0. 基础配置
# ==============================
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../data/classifier_dataset_hsv"))
SAVE_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../result/checkpoints"))
MODEL_NAME = "best_swin_dca_only.pth"
CLASS_INDEX_NAME = "class_indices_swin_dca.json"

MODEL_ID = "swin_base_patch4_window7_224.ms_in22k_ft_in1k"
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
INIT_DCA_GAMMA = 0.0

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
# 1. 核心模块：通道级动态残差路由 (Channel-wise Dynamic Residual Routing)
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
        # 修复点 1：必须先执行 downsample
        if hasattr(self.orig_stage, 'downsample') and self.orig_stage.downsample is not None:
            x = self.orig_stage.downsample(x)

        anchors = []

        for i, block in enumerate(self.orig_stage.blocks):
            next_x = block(x)

            if i in self.target_indices:
                gates = self.routers[str(i)](next_x)
                gamma = self.gammas[str(i)]

                routed_feat = torch.zeros_like(next_x)
                for j, anchor_feat in enumerate(anchors):
                    routed_feat = routed_feat + gates[:, j] * anchor_feat

                next_x = next_x + gamma * routed_feat

            x = next_x

            if i in self.anchor_indices:
                anchors.append(x)

        return x

def inject_dynamic_residual_routing(model):
    print("正在向 Stage 3 注入动态残差路由 (Dynamic Residual Routing)...")
    orig_stage_3 = model.layers[2]

    wrapper = DynamicResidualStageWrapper(
        orig_stage=orig_stage_3,
        anchor_indices=(1, 4, 9),
        target_indices=(11, 14, 17),
        dim=512
    )

    model.layers[2] = wrapper
    print("注入完成: anchor block [1, 4, 9], target block [11, 14, 17]")
    return model

# ==============================
# 3. 辅助函数与训练逻辑
# ==============================
def build_transforms(image_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf

def build_dataloaders(data_dir, image_size=224, batch_size=32, num_workers=8):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_tf, val_tf = build_transforms(image_size)
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=train_tf),
        'val': datasets.ImageFolder(val_dir, transform=val_tf),
    }

    dataloaders = {
        'train': DataLoader(
            image_datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(DEVICE.type == 'cuda'),
        ),
        'val': DataLoader(
            image_datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(DEVICE.type == 'cuda'),
        ),
    }
    return image_datasets, dataloaders

def get_model(num_classes):
    print(f"正在加载原始预训练 Swin-Base: {MODEL_ID}")
    model = timm.create_model(MODEL_ID, pretrained=True, num_classes=num_classes)
    model = inject_dynamic_residual_routing(model)
    return model

def build_optimizer(model):
    # 修复点 3：分类头 head 加入大学习率组
    fast_keywords = get_fast_keywords()

    backbone_params = []
    fast_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in fast_keywords):
            fast_params.append(param)
        else:
            backbone_params.append(param)

    print(f"Backbone params groups: {len(backbone_params)} tensors | lr={LR_BACKBONE}")
    print(f"New Routing/Head params groups: {len(fast_params)} tensors | lr={LR_NEW}")

    optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': LR_BACKBONE},
            {'params': fast_params, 'lr': LR_NEW},
        ], weight_decay=WEIGHT_DECAY)
    return optimizer

def build_scheduler(optimizer, total_epochs, warmup_epochs=5, freeze_epochs=3):
    # 双路调度，配合物理冻结更加稳妥
    def lr_lambda_backbone(epoch):
        if epoch < freeze_epochs:
            return 0.0
        adj_epoch = epoch - freeze_epochs
        if adj_epoch < warmup_epochs:
            return max(0.05, adj_epoch / max(1, warmup_epochs))
        progress = float(adj_epoch - warmup_epochs) / float(max(1, total_epochs - freeze_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def lr_lambda_new(epoch):
        if epoch < warmup_epochs:
            return max(0.05, epoch / max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, [lr_lambda_backbone, lr_lambda_new])

def compute_class_weights(dataset):
    targets = np.array(dataset.targets)
    num_classes = len(dataset.classes)
    counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)

def get_fast_keywords():
    return ['routers.', 'gammas.', 'head.']

# 修复点 2：真正的物理冻结/解冻函数
def set_backbone_trainable(model, trainable=True):
    fast_keywords = get_fast_keywords()
    for name, param in model.named_parameters():
        if not any(k in name for k in fast_keywords):
            param.requires_grad = trainable

def run_one_epoch(
    model,
    loader,
    criterion,
    num_classes,
    optimizer=None,
    scaler=None,
    fast_keys=None,
    is_freeze=False,
):
    is_train = optimizer is not None
    if is_train:
        if is_freeze:
            model.eval()
            fast_keys = fast_keys or []
            for name, module in model.named_modules():
                if any(k in name for k in fast_keys):
                    module.train()
        else:
            model.train()
    else:
        model.eval()

    running_loss = 0.0
    preds_all = []
    labels_all = []

    desc_str = "Train" if is_train else "Val"
    pbar = tqdm(loader, desc=desc_str, leave=False)

    for inputs, labels in pbar:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)

            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                active_params = [p for p in model.parameters() if p.requires_grad]
                torch.nn.utils.clip_grad_norm_(active_params, max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

        running_loss += loss.item() * inputs.size(0)
        preds_all.extend(preds.detach().cpu().numpy().tolist())
        labels_all.extend(labels.detach().cpu().numpy().tolist())

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    epoch_loss = running_loss / max(1, len(loader.dataset))
    epoch_acc = 100.0 * (np.array(preds_all) == np.array(labels_all)).mean()
    epoch_f1 = 100.0 * f1_score(
        labels_all,
        preds_all,
        labels=list(range(num_classes)),
        average='macro',
        zero_division=0,
    )
    return epoch_loss, epoch_acc, epoch_f1

def main():
    set_seed(SEED)
    gc.collect()
    torch.cuda.empty_cache()
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"DATA_DIR: {DATA_DIR}")

    image_datasets, dataloaders = build_dataloaders(
        DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    num_classes = len(image_datasets['train'].classes)

    model = get_model(num_classes).to(DEVICE)
    class_weights = compute_class_weights(image_datasets['train'])
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(
        optimizer,
        total_epochs=NUM_EPOCHS,
        warmup_epochs=WARMUP_EPOCHS,
        freeze_epochs=FREEZE_EPOCHS,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    best_val_f1 = -1.0
    best_model_wts = copy.deepcopy(model.state_dict())
    save_path = os.path.join(SAVE_DIR, MODEL_NAME)

    since = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")
        print("-" * 60)

        # 控制主干的冻结状态
        if epoch < FREEZE_EPOCHS:
            if epoch == 0:
                print("[Epoch 0-2] 物理冻结主干网络，训练动态残差路由...")
            set_backbone_trainable(model, trainable=False)
        elif epoch == FREEZE_EPOCHS:
            print("[Epoch 3] 解冻主干网络，开始全量微调。")
            set_backbone_trainable(model, trainable=True)

        train_loss, train_acc, train_f1 = run_one_epoch(
            model,
            dataloaders['train'],
            criterion,
            num_classes,
            optimizer,
            scaler,
            fast_keys=get_fast_keywords(),
            is_freeze=epoch < FREEZE_EPOCHS,
        )
        val_loss, val_acc, val_f1 = run_one_epoch(model, dataloaders['val'], criterion, num_classes, None, None)

        scheduler.step()

        lr_backbone = optimizer.param_groups[0]['lr']
        lr_new = optimizer.param_groups[1]['lr']

        print(
            f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | Macro-F1: {train_f1:.2f}%\n"
            f"Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | Macro-F1: {val_f1:.2f}%\n"
            f"LR    | backbone: {lr_backbone:.7f} | new/head: {lr_new:.7f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, save_path)
            print(f"保存当前最佳模型: {save_path}")

    elapsed = time.time() - since
    print(f"\nTraining finished in {elapsed / 60:.1f} min")
    print(f"Best Val Macro-F1: {best_val_f1:.2f}%")

    with open(os.path.join(SAVE_DIR, CLASS_INDEX_NAME), 'w', encoding='utf-8') as f:
        json.dump(image_datasets['train'].class_to_idx, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
