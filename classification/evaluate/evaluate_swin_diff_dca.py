# -*- coding: utf-8 -*-
import os
import json
import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

import timm
from timm.layers import to_2tuple, trunc_normal_
from evaluate_report_utils import (
    format_eval_report,
    save_confusion_matrix_counts,
    write_report,
)

warnings.filterwarnings("ignore")

# ==============================================================================
# 0. 基础配置 (🌟 修复：自洽的绝对路径)
# ==============================================================================
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../data/classifier_dataset_hsv/test")) 
CHECKPOINT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../result/checkpoints"))
WEIGHT_PATH = os.path.join(CHECKPOINT_DIR, "best_swin_diff_dca_aligned.pth")
CLASS_INDEX_PATH = os.path.join(CHECKPOINT_DIR, "class_indices_swin_diff_dca.json")
EVAL_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(BASE_DIR, EVAL_NAME)

MODEL_ID = "swin_base_patch4_window7_224.ms_in22k_ft_in1k"
BATCH_SIZE = 32 
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 224
TARGET_STAGE_DIMS = (1024,) 
LITE_RATIO = 0.25               
INIT_LAMBDA = -2.0               
INIT_DIFF_GAMMA = 0.02
INIT_DCA_GAMMA = 0.0  

def save_evaluation_results(
    report_lines,
    results,
    summary,
    all_labels,
    all_preds,
    dataset,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_report(report_lines, OUTPUT_DIR)
    print(f"💾 report.txt 已保存至: {OUTPUT_DIR}")

def save_confusion_matrices(all_labels, all_preds, class_names):
    save_confusion_matrix_counts(
        all_labels,
        all_preds,
        class_names,
        OUTPUT_DIR,
        file_name="confusion_matrix_swin_diff_dca_counts.png",
        model_name="swin_diff_dca",
    )

# ==============================================================================
# 1. 核心架构：差分注意力 (Diff) -> 必须与训练端 1:1 对齐
# ==============================================================================
class DifferentialWindowAttention(nn.Module):
    fused_attn: torch.jit.Final[bool]

    def __init__(
        self, dim: int, num_heads: int, head_dim: Optional[int] = None,
        window_size: tuple = 7, qkv_bias: bool = True,
        attn_drop: float = 0., proj_drop: float = 0.,
        device=None, dtype=None,
    ):
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.scale = head_dim ** -0.5
        self.fused_attn = False

        self.relative_position_bias_table = nn.Parameter(
            torch.empty((2 * win_h - 1) * (2 * win_w - 1), num_heads, **dd)
        )
        self.register_buffer(
            "relative_position_index",
            torch.empty(win_h * win_w, win_h * win_w, device=device, dtype=torch.long),
            persistent=False,
        )

        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias, **dd)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim, **dd)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.use_diff = dim in TARGET_STAGE_DIMS

        if self.use_diff:
            head_dim_lite = max(1, int(head_dim * LITE_RATIO))
            self.lite_dim = head_dim_lite * num_heads

            self.q_lite = nn.Linear(dim, self.lite_dim, bias=False, **dd)
            self.k_lite = nn.Linear(dim, self.lite_dim, bias=False, **dd)
            self.v_lite = nn.Linear(dim, self.lite_dim, bias=False, **dd)

            self.lambda_1_raw = nn.Parameter(torch.tensor(float(INIT_LAMBDA), **dd))
            self.lambda_2_raw = nn.Parameter(torch.tensor(float(INIT_LAMBDA), **dd))

            self.diff_proj = nn.Linear(self.lite_dim, dim, **dd)
            trunc_normal_(self.diff_proj.weight, std=1e-4)
            nn.init.zeros_(self.diff_proj.bias)

            self.diff_gamma = nn.Parameter(torch.tensor(float(INIT_DIFF_GAMMA), **dd))

        if not self.proj.weight.is_meta:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self._init_buffers()

    def _init_buffers(self) -> None:
        win_h, win_w = self.window_size
        from timm.models.swin_transformer import get_relative_position_index
        self.relative_position_index.copy_(
            get_relative_position_index(win_h, win_w, device=self.relative_position_index.device)
        )

    def set_window_size(self, window_size: Tuple[int, int]) -> None:
        window_size = to_2tuple(window_size)
        if window_size == self.window_size: return
        self.window_size = window_size
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        with torch.no_grad():
            new_bias_shape = (2 * win_h - 1) * (2 * win_w - 1), self.num_heads
            from timm.models.swin_transformer import resize_rel_pos_bias_table, get_relative_position_index
            self.relative_position_bias_table = nn.Parameter(
                resize_rel_pos_bias_table(
                    self.relative_position_bias_table,
                    new_window_size=self.window_size,
                    new_bias_shape=new_bias_shape,
                )
            )
            self.register_buffer(
                "relative_position_index",
                get_relative_position_index(win_h, win_w, device=self.relative_position_bias_table.device),
                persistent=False,
            )

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_area, self.window_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, _ = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        rel_pos_bias = self._get_rel_pos_bias()
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn + rel_pos_bias

        if mask is not None:
            num_win = mask.shape[0]
            attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn_orig_softmax = self.softmax(attn)
        attn_orig_drop = self.attn_drop(attn_orig_softmax)
        x_orig = attn_orig_drop @ v

        if not self.use_diff:
            x_out = x_orig.transpose(1, 2).reshape(B_, N, -1)
            x_out = self.proj(x_out)
            x_out = self.proj_drop(x_out)
            return x_out

        head_dim_lite = self.lite_dim // self.num_heads
        scale_lite = head_dim_lite ** -0.5

        q_l = self.q_lite(x).reshape(B_, N, self.num_heads, head_dim_lite).permute(0, 2, 1, 3)
        k_l = self.k_lite(x).reshape(B_, N, self.num_heads, head_dim_lite).permute(0, 2, 1, 3)
        v_l = self.v_lite(x).reshape(B_, N, self.num_heads, head_dim_lite).permute(0, 2, 1, 3)

        attn_lite = (q_l * scale_lite) @ k_l.transpose(-2, -1)
        attn_lite = attn_lite + rel_pos_bias

        if mask is not None:
            attn_lite = attn_lite.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn_lite = attn_lite.view(-1, self.num_heads, N, N)

        attn_lite_softmax = self.softmax(attn_lite)

        lam1 = torch.sigmoid(self.lambda_1_raw)
        lam2 = torch.sigmoid(self.lambda_2_raw)

        diff_attn = lam1 * attn_lite_softmax - lam2 * attn_orig_softmax.detach()
        diff_attn = self.attn_drop(diff_attn)
        delta_v = diff_attn @ v_l

        x_orig = x_orig.transpose(1, 2).reshape(B_, N, -1)
        delta_v = delta_v.transpose(1, 2).reshape(B_, N, self.lite_dim)

        x_out = x_orig + self.diff_gamma * self.diff_proj(delta_v)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out

def inject_stage4_diff_attention(model):
    """与训练脚本一致：只在 Stage 4 原位替换窗口注意力，避免全局 monkey patch。"""
    layer = model.layers[3]
    for block in layer.blocks:
        old = block.attn
        new = DifferentialWindowAttention(
            dim=old.qkv.in_features,
            num_heads=old.num_heads,
            window_size=old.window_size,
            qkv_bias=old.qkv.bias is not None,
        )
        new.load_state_dict(old.state_dict(), strict=False)
        new.relative_position_index.copy_(old.relative_position_index)
        block.attn = new.to(old.qkv.weight.device)
    return model

# ==============================================================================
# 2. 核心架构：动态残差路由 (DCA) -> 必须与训练端 1:1 对齐
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
        B = x.shape[0]
        C = x.shape[-1]
        x_pooled = x.reshape(B, -1, C).mean(dim=1)               
        gates = self.fc(x_pooled)           
        gates = gates.reshape(B, self.num_anchors, C)            
        gates = torch.softmax(gates, dim=1)                      
        broadcast_shape = [B, self.num_anchors] + [1] * (x.dim() - 2) + [C]
        gates = gates.reshape(*broadcast_shape)
        return gates

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
        if hasattr(self.orig_stage, 'downsample') and self.orig_stage.downsample is not None:
            x = self.orig_stage.downsample(x)

        anchors = []
        for i, block in enumerate(self.orig_stage.blocks):
            next_x = block(x)

            if i in self.target_indices:
                router = self.routers[str(i)]
                gamma = self.gammas[str(i)]
                gates = router(next_x)

                routed_feat = torch.zeros_like(next_x)
                for j, anchor_feat in enumerate(anchors):
                    routed_feat = routed_feat + gates[:, j] * anchor_feat

                next_x = next_x + gamma * routed_feat

            x = next_x

            if i in self.anchor_indices:
                anchors.append(x)

        return x

def inject_dynamic_residual_routing(model):
    orig_stage_3 = model.layers[2]
    wrapper = DynamicResidualStageWrapper(
        orig_stage=orig_stage_3,
        anchor_indices=(1, 4, 9),
        target_indices=(11, 14, 17), 
        dim=512,
    )
    model.layers[2] = wrapper
    return model

def get_model(num_classes):
    print(f"🏗️ 构建 Swin-Diff-DCA 架构以加载权重...")
    model = timm.create_model(MODEL_ID, pretrained=False, num_classes=num_classes)
    model = inject_dynamic_residual_routing(model)
    model = inject_stage4_diff_attention(model)
    return model

def normalize_dca_checkpoint_for_model(checkpoint, model):
    """兼容旧版 DCA 权重命名：fc1/fc2、dca_gammas -> 当前 fc.0/fc.2、gammas。"""
    normalized = {}
    for key, value in checkpoint.items():
        new_key = key
        if ".routers." in new_key:
            new_key = new_key.replace(".fc1.", ".fc.0.")
            new_key = new_key.replace(".fc2.", ".fc.2.")
        new_key = new_key.replace(".dca_gammas.", ".gammas.")
        normalized[new_key] = value

    model_state = model.state_dict()
    for key, value in model_state.items():
        if key.startswith("layers.2.gammas.") and key not in normalized:
            normalized[key] = value
    return normalized

# ==============================================================================
# 3. 评估主程序
# ==============================================================================
def main():
    if not os.path.exists(DATA_DIR):
        print(f"❌ 错误: 测试集路径不存在 {DATA_DIR}")
        return
    if not os.path.exists(WEIGHT_PATH):
        print(f"❌ 错误: 权重文件不存在 {WEIGHT_PATH}")
        return

    # 测试集的数据预处理
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=test_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    class_names = dataset.classes
    num_classes = len(class_names)

    if os.path.exists(CLASS_INDEX_PATH):
        with open(CLASS_INDEX_PATH, "r", encoding="utf-8") as f:
            saved_class_to_idx = json.load(f)
        if dataset.class_to_idx != saved_class_to_idx:
            print("❌ 错误: 测试集类别映射与训练时保存的 class_indices 不一致。")
            print(f"   dataset.class_to_idx: {dataset.class_to_idx}")
            print(f"   saved_class_to_idx  : {saved_class_to_idx}")
            return
    
    print("\n" + "="*80)
    print(f"🏁 开始评估:【Swin + Diff + DCA 终极版】")
    print(f"✅ 数据集类别数: {num_classes} | 纯净测试集样本总数: {len(dataset)}")
    print(f"📥 正在加载权重: {WEIGHT_PATH}")
    print(f"📁 结果输出目录: {OUTPUT_DIR}")

    model = get_model(num_classes=num_classes)
    
    try:
        checkpoint = torch.load(WEIGHT_PATH, map_location=DEVICE)
        # 兼容 DDP 前缀
        if list(checkpoint.keys())[0].startswith('module.'):
            checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        checkpoint = normalize_dca_checkpoint_for_model(checkpoint, model)
        model.load_state_dict(checkpoint, strict=True)
        print("✅ 权重完美加载！")
    except Exception as e:
        print(f"❌ 权重加载失败，错误详情: {e}")
        return

    model.to(DEVICE)
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction='none')

    all_preds = []
    all_labels = []
    class_loss_sum = {i: 0.0 for i in range(num_classes)}
    class_count = {i: 0 for i in range(num_classes)}

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="正在全量扫描测试集"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            losses = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(images.size(0)):
                label_idx = labels[i].item()
                loss_val = losses[i].item()
                class_loss_sum[label_idx] += loss_val
                class_count[label_idx] += 1

    # 🌟 核心修复：显式指定 labels=range(num_classes)，防止缺省类别导致的 Macro 虚高
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(num_classes), zero_division=0
    )
    
    results = []
    for i, name in enumerate(class_names):
        avg_loss = class_loss_sum[i] / class_count[i] if class_count[i] > 0 else 0.0
        results.append({
            "name": name,
            "support": support[i],
            "p": precision[i],
            "r": recall[i],
            "f1": f1[i],
            "loss": avg_loss
        })

    # 按 F1 分数从小到大排序，方便查看长尾类表现
    results.sort(key=lambda x: x['f1'], reverse=False)

    avg_p = np.average(precision, weights=support)
    avg_r = np.average(recall, weights=support)
    avg_f1 = np.average(f1, weights=support)
    macro_f1 = np.mean(f1)
    
    total_loss = sum(class_loss_sum.values())
    total_count = sum(class_count.values())
    global_avg_loss = total_loss / total_count if total_count > 0 else 0
    accuracy = float(np.mean(np.array(all_labels) == np.array(all_preds))) if all_labels else 0.0
    report_lines = format_eval_report(
        model_name="swin_diff_dca",
        data_dir=DATA_DIR,
        weight_path=WEIGHT_PATH,
        class_index_path=CLASS_INDEX_PATH,
        num_classes=num_classes,
        num_samples=len(dataset),
        accuracy=accuracy,
        macro_precision=float(np.mean(precision)),
        macro_recall=float(np.mean(recall)),
        macro_f1=float(macro_f1),
        weighted_precision=float(avg_p),
        weighted_recall=float(avg_r),
        weighted_f1=float(avg_f1),
        avg_loss=float(global_avg_loss),
        rows=results,
    )
    output_text = "\n".join(report_lines)
    print(output_text + "\n")

    summary = {
        "model": "swin_diff_dca",
        "data_dir": DATA_DIR,
        "weight_path": WEIGHT_PATH,
        "class_index_path": CLASS_INDEX_PATH,
        "num_classes": num_classes,
        "num_samples": len(dataset),
        "weighted_precision": float(avg_p),
        "weighted_recall": float(avg_r),
        "weighted_f1": float(avg_f1),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(macro_f1),
        "avg_loss": float(global_avg_loss),
    }
    save_evaluation_results(report_lines, results, summary, all_labels, all_preds, dataset)
    save_confusion_matrices(all_labels, all_preds, class_names)

if __name__ == "__main__":
    main()
