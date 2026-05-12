import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm

import timm
from typing import Optional, Tuple
from timm.layers import to_2tuple, trunc_normal_

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ==============================================================================
# 0. 配置区域 
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../result/checkpoints"))
WEIGHT_PATH = os.path.join(CHECKPOINT_DIR, "best_swin_diff.pth")
MODEL_ARCH = "swin_base_patch4_window7_224.ms_in22k_ft_in1k" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 保持与训练完全一致
TARGET_STAGE_DIMS = (512, 1024) 
LITE_RATIO = 0.25               
INIT_LAMBDA = 0.0               

# ==============================================================================
# 1. 核心魔改：2.0 版差分注意力补丁
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
            torch.empty((2 * win_h - 1) * (2 * win_w - 1), num_heads, **dd))
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
            self.diff_gamma = nn.Parameter(torch.tensor(0.01, **dd))
            nn.init.zeros_(self.diff_proj.weight)
            nn.init.zeros_(self.diff_proj.bias)

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
        pass

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1) 
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() 
        return relative_position_bias.unsqueeze(0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
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
            x = x_orig.transpose(1, 2).reshape(B_, N, -1)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

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

        x = x_orig + self.diff_gamma * self.diff_proj(delta_v)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# 打补丁！
timm.models.swin_transformer.WindowAttention = DifferentialWindowAttention

# ==============================================================================
# 2. Grad-CAM 专用辅助函数
# ==============================================================================
def reshape_transform(tensor, height=7, width=7):
    result = tensor.permute(0, 3, 1, 2)
    return result

def load_image(img_path):
    rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(rgb_img).unsqueeze(0).to(DEVICE)
    return rgb_img, input_tensor

# ==============================================================================
# 3. 批量处理主函数
# ==============================================================================
def main(input_dir, output_dir):
    print("🏗️ 正在加载 Swin-Diff 架构并注入 88.50% 的王者权重...")
    model = timm.create_model(MODEL_ARCH, pretrained=False, num_classes=25)
    
    checkpoint = torch.load(WEIGHT_PATH, map_location=DEVICE)
    if list(checkpoint.keys())[0].startswith('module.'):
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=True)
    model.to(DEVICE)
    model.eval()

    # 选取最后一层的 norm1 作为目标提取层
    target_layers = [model.layers[-1].blocks[-1].norm1]
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    
    # 筛选有效的图片文件
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    img_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
    
    if len(img_files) == 0:
        print(f"❌ 错误：在文件夹 {input_dir} 中没有找到任何图片！")
        return

    print(f"🔍 找到 {len(img_files)} 张图片，开始批量生成热力图...")
    
    for filename in tqdm(img_files, desc="生成进度"):
        img_path = os.path.join(input_dir, filename)
        
        # 1. 读取并预处理
        rgb_img, input_tensor = load_image(img_path)

        # 2. 生成热力图
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # 3. 绘图拼接
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(rgb_img)
        axes[0].set_title('Original Soybean Image')
        axes[0].axis('off')

        axes[1].imshow(cam_image)
        axes[1].set_title('Swin-Diff Grad-CAM')
        axes[1].axis('off')

        # 4. 保存图片并防内存泄漏
        save_path = os.path.join(output_dir, f"cam_{filename}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig) # 极其重要：画完立刻清理内存！

    print(f"✅ 批量生成完毕！所有对比图已保存在: {output_dir}")

if __name__ == "__main__":
    # ================= 配置输入和输出文件夹 =================
    # 把这里的 INPUT_DIR 换成你想跑的那个类别文件夹
    INPUT_DIR = "../../data/classifier_dataset_hsv/test/zd59" 
    
    # 生成的图片会存放在这个新文件夹里
    OUTPUT_DIR = "./cam_results_zd59" 
    # ========================================================
    
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 找不到输入文件夹: {INPUT_DIR}")
    else:
        main(INPUT_DIR, OUTPUT_DIR)
