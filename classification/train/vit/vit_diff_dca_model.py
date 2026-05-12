# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


LITE_RATIO = 0.25
INIT_LAMBDA = -2.0
INIT_DIFF_GAMMA = 0.02
INIT_DCA_GAMMA = 0.0


class DifferentialSelfAttention(nn.Module):
    """在 ViT 深层 self-attention 上添加轻量差分注意力分支，保留原 MHA 预训练权重。"""

    def __init__(self, base_attn):
        super().__init__()
        self.base_attn = base_attn
        self.embed_dim = base_attn.embed_dim
        self.num_heads = base_attn.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        lite_head_dim = max(1, int(self.head_dim * LITE_RATIO))
        self.lite_dim = lite_head_dim * self.num_heads
        self.q_lite = nn.Linear(self.embed_dim, self.lite_dim, bias=False)
        self.k_lite = nn.Linear(self.embed_dim, self.lite_dim, bias=False)
        self.v_lite = nn.Linear(self.embed_dim, self.lite_dim, bias=False)
        self.diff_proj = nn.Linear(self.lite_dim, self.embed_dim)
        self.diff_drop = nn.Dropout(base_attn.dropout)
        self.diff_gamma = nn.Parameter(torch.tensor(float(INIT_DIFF_GAMMA)))
        self.lambda_1_raw = nn.Parameter(torch.tensor(float(INIT_LAMBDA)))
        self.lambda_2_raw = nn.Parameter(torch.tensor(float(INIT_LAMBDA)))

        nn.init.trunc_normal_(self.diff_proj.weight, std=1e-4)
        nn.init.zeros_(self.diff_proj.bias)

    def _project_base_qk(self, x):
        weight = self.base_attn.in_proj_weight
        bias = self.base_attn.in_proj_bias
        q_weight = weight[:self.embed_dim]
        k_weight = weight[self.embed_dim: 2 * self.embed_dim]
        q_bias = bias[:self.embed_dim] if bias is not None else None
        k_bias = bias[self.embed_dim: 2 * self.embed_dim] if bias is not None else None

        q = F.linear(x, q_weight, q_bias)
        k = F.linear(x, k_weight, k_bias)
        q = q.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        return q, k

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        base_out, base_weights = self.base_attn(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

        if query is not key or key is not value:
            return base_out, base_weights

        q_base, k_base = self._project_base_qk(query)
        base_attn = (q_base * self.scale) @ k_base.transpose(-2, -1)

        lite_head_dim = self.lite_dim // self.num_heads
        q_lite = self.q_lite(query).reshape(query.shape[0], query.shape[1], self.num_heads, lite_head_dim).transpose(1, 2)
        k_lite = self.k_lite(query).reshape(query.shape[0], query.shape[1], self.num_heads, lite_head_dim).transpose(1, 2)
        v_lite = self.v_lite(query).reshape(query.shape[0], query.shape[1], self.num_heads, lite_head_dim).transpose(1, 2)

        lite_attn = (q_lite * (lite_head_dim ** -0.5)) @ k_lite.transpose(-2, -1)
        lam1 = torch.sigmoid(self.lambda_1_raw)
        lam2 = torch.sigmoid(self.lambda_2_raw)
        diff_attn = lam1 * torch.softmax(lite_attn, dim=-1) - lam2 * torch.softmax(base_attn.detach(), dim=-1)
        diff_out = (self.diff_drop(diff_attn) @ v_lite).transpose(1, 2).reshape(query.shape[0], query.shape[1], self.lite_dim)
        base_out = base_out + self.diff_gamma * self.diff_proj(diff_out)
        return base_out, base_weights


class ChannelGating(nn.Module):
    def __init__(self, dim, num_anchors):
        super().__init__()
        hidden_dim = max(32, dim // 4)
        self.fc = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_anchors * dim),
        )
        self.num_anchors = num_anchors
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x):
        pooled = x.mean(dim=1)
        gates = torch.softmax(self.fc(pooled).reshape(x.shape[0], self.num_anchors, x.shape[-1]), dim=1)
        return gates


class DCAViTEncoderLayers(nn.Module):
    """在 ViT 后段 encoder block 之间做动态通道路由，不改变 patch embedding 和分类头。"""

    def __init__(self, orig_layers, anchor_idx=(2, 5, 8), target_idx=(9, 10, 11), dim=768):
        super().__init__()
        self.orig_layers = orig_layers
        self.anchor_idx = anchor_idx
        self.target_idx = target_idx
        self.routers = nn.ModuleDict({str(i): ChannelGating(dim, len(anchor_idx)) for i in target_idx})
        self.gammas = nn.ParameterDict({str(i): nn.Parameter(torch.tensor(float(INIT_DCA_GAMMA))) for i in target_idx})

    def forward(self, x):
        anchors = []
        for idx, layer in enumerate(self.orig_layers):
            nx = layer(x)
            if idx in self.target_idx and anchors:
                gates = self.routers[str(idx)](nx)
                routed = 0
                for anchor_i, anchor_feat in enumerate(anchors):
                    routed = routed + gates[:, anchor_i].unsqueeze(1) * anchor_feat
                nx = nx + self.gammas[str(idx)] * routed
            x = nx
            if idx in self.anchor_idx:
                anchors.append(x)
        return x


def inject_vit_diff_dca(model):
    print("实例级注入 ViT Diff Attention 与 DCA 模块...")

    # ViT-B/16 共有 12 个 encoder block。为保持与 Swin 主线一致，只在深层注入 Diff 分支。
    for idx in (9, 10, 11):
        block = model.encoder.layers[idx]
        block.self_attention = DifferentialSelfAttention(block.self_attention)

    # ViT 没有层级 stage，这里用浅/中层 block 作为 anchor，深层 block 作为 target。
    model.encoder.layers = DCAViTEncoderLayers(
        model.encoder.layers,
        anchor_idx=(2, 5, 8),
        target_idx=(9, 10, 11),
        dim=model.hidden_dim,
    )
    return model


def get_vit_diff_dca_model(num_classes, pretrained=True):
    if pretrained:
        print("正在通过 torchvision 加载 ViT_B_16 (ImageNet-1K 预训练底座) ...")
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.vit_b_16(weights=weights)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return inject_vit_diff_dca(model)


def get_fast_keywords():
    return [
        "heads.head.",
        "q_lite",
        "k_lite",
        "v_lite",
        "diff_proj",
        "diff_gamma",
        "lambda_",
        "routers.",
        "gammas.",
    ]
