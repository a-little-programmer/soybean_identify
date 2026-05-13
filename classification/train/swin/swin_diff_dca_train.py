# -*- coding: utf-8 -*-
import os
import gc
import json
import time
import math
import random
import warnings
from typing import Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from sklearn.metrics import f1_score

import timm
from timm.layers import to_2tuple, trunc_normal_

warnings.filterwarnings("ignore")

# ==============================
# 0. 基础配置 (消融实验对齐参数)
# ==============================
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../../data/classifier_dataset_hsv"))
SAVE_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../result/checkpoints"))
MODEL_NAME = "best_swin_diff_dca_aligned.pth"
CLASS_INDEX_NAME = "class_indices_swin_diff_dca.json"

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

TARGET_STAGE_DIMS = (1024,)
LITE_RATIO = 0.25
INIT_LAMBDA = -2.0
INIT_DIFF_GAMMA = 0.02
INIT_DCA_GAMMA = 0.0
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==============================================================================
# 1. 核心架构逻辑 (Diff Attention & Dynamic Channel Routing)
# ==============================================================================
class DifferentialWindowAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        qkv_bias=True,
        attn_drop=0.,
        proj_drop=0.,
        device=None,
        dtype=None,
    ):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.dim, self.num_heads = dim, num_heads
        self.window_size = to_2tuple(window_size)
        self.window_area = self.window_size[0] * self.window_size[1]
        self.scale = (dim // num_heads) ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.empty(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                num_heads,
                **dd,
            )
        )
        self.register_buffer(
            "relative_position_index",
            torch.empty(self.window_area, self.window_area, device=device, dtype=torch.long),
            persistent=False,
        )
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
        return (
            self.relative_position_bias_table[self.relative_position_index.view(-1)]
            .view(self.window_area, self.window_area, -1)
            .permute(2, 0, 1)
            .contiguous()
            .unsqueeze(0)
        )

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

class DCAWrapper(nn.Module):
    def __init__(self, orig_stage, anchor_idx=(1, 4, 9), target_idx=(11, 14, 17), dim=512):
        super().__init__()
        self.orig_stage, self.anchor_idx, self.target_idx = orig_stage, anchor_idx, target_idx
        self.routers = nn.ModuleDict({str(i): ChannelGating(dim, len(anchor_idx)) for i in target_idx})
        self.gammas = nn.ParameterDict({str(i): nn.Parameter(torch.tensor(float(INIT_DCA_GAMMA))) for i in target_idx})

    def forward(self, x):
        if hasattr(self.orig_stage, 'downsample') and self.orig_stage.downsample is not None:
            x = self.orig_stage.downsample(x)
        anchors = []
        for i, block in enumerate(self.orig_stage.blocks):
            nx = block(x)
            if i in self.target_idx:
                g = self.routers[str(i)](nx)
                feat = 0
                for j in range(len(anchors)):
                    feat = feat + g[:, j] * anchors[j]
                gamma = self.gammas[str(i)]
                nx = nx + gamma * feat
            x = nx
            if i in self.anchor_idx:
                anchors.append(x)
        return x

def inject_all(model):
    print("实例级注入 Diff 与 DCA 模块...")
    # 1. 仅注入 DCA 到 Stage 3
    model.layers[2] = DCAWrapper(
        model.layers[2],
        anchor_idx=(1, 4, 9),
        target_idx=(11, 14, 17),
        dim=512,
    )

    # 2. 仅注入 Diff 到 Stage 4 (无需浪费性能替换前置 Stage)
    layer = model.layers[3]
    for block in layer.blocks:
        old = block.attn
        new = DifferentialWindowAttention(
            old.qkv.in_features,
            old.num_heads,
            old.window_size,
            old.qkv.bias is not None,
        )
        new.load_state_dict(old.state_dict(), strict=False)
        new.relative_position_index.copy_(old.relative_position_index)
        block.attn = new.to(old.qkv.weight.device)
    return model

# ==============================================================================
# 2. 数据加载与训练工具
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
    return ['q_lite', 'k_lite', 'v_lite', 'diff', 'lambda', 'routers.', 'gammas.', 'head.']

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

    model = inject_all(timm.create_model(MODEL_ID, pretrained=True, num_classes=num_classes)).to(DEVICE)

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

        if is_freeze and epoch == 0:
            print("开启物理冻结...")
            for n, p in model.named_parameters():
                if not any(k in n for k in fast_keys):
                    p.requires_grad = False
        elif epoch == FREEZE_EPOCHS:
            print("解冻 Backbone...")
            for p in model.parameters():
                p.requires_grad = True

        for phase in ['train', 'val']:
            is_train = phase == 'train'
            if is_train:
                if is_freeze:
                    model.eval()
                    for n, m in model.named_modules():
                        if any(k in n for k in fast_keys):
                            m.train()
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
                        # 只裁剪活跃梯度的范数，防止冻结参数污染
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
