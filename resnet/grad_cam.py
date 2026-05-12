import os
import cv2
import json
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
import timm

from evaluate_swin_dca import (
    inject_dynamic_residual_routing,
    normalize_dca_checkpoint_for_model as normalize_dca_checkpoint,
)
from evaluate_swin_diff import inject_only_diff
from evaluate_swin_diff_dca import (
    get_model as get_swin_diff_dca_model,
    normalize_dca_checkpoint_for_model as normalize_diff_dca_checkpoint,
)

# 导入 Grad-CAM 相关的包
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ================= 🌟 核心配置区域 =================
# 1. 想测哪个模型？就在这里改！
# 可选值: 'resnet50', 'regnet', 'swin', 'vit',
#        'swin_aligned', 'swin_diff', 'swin_dca_only', 'swin_diff_dca'
MODEL_TYPE = {"swin_diff", "swin_dca_only", "swin_diff_dca"}
# 2. 待测试的【大文件夹】路径 (程序会自动遍历里面所有的小文件夹)
IMAGE_DIR = "/nfs/spy/soybean_detect/data/classifier_dataset_hsv/train"


# 4. 权重与类别索引路径 (请确保您的权重文件名对应如下格式，如果不一致请手动修改)
CHECKPOINT_PATHS = {
    "resnet50": "checkpoints/best_resnet50_soybean.pth",
    "regnet": "checkpoints/best_regnet_soybean.pth",
    "swin": "checkpoints/best_swin_soybean.pth",
    "vit": "checkpoints/best_vit_soybean.pth",
    "swin_aligned": "checkpoints/best_swin_aligned.pth",
    "swin_diff": "checkpoints/best_swin_diff.pth",
    "swin_dca_only": "checkpoints/best_swin_dca_only.pth",
    "swin_diff_dca": "checkpoints/best_swin_diff_dca_aligned.pth",
}
CLASS_IDX_PATH = "checkpoints/class_indices_regnet.json"

TIMM_SWIN_ID = "swin_base_patch4_window7_224.ms_in22k_ft_in1k"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =================================================

def load_checkpoint(model, model_path, normalize_fn=None):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if list(checkpoint.keys())[0].startswith("module."):
        checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    if normalize_fn is not None:
        checkpoint = normalize_fn(checkpoint, model)
    model.load_state_dict(checkpoint, strict=True)

def reshape_transform_swin(tensor):
    # timm Swin 输出是 channels-last: [B, H, W, C]，Grad-CAM 需要 [B, C, H, W]。
    return tensor.permute(0, 3, 1, 2)

def setup_model_and_cam(modeltype, num_classes):
    """根据选择的模型类型，动态初始化网络架构和 Grad-CAM 目标层"""
    print(f"🏗️ 正在初始化 {modeltype.upper()} 模型环境...")
    model_path = CHECKPOINT_PATHS[modeltype]
    reshape_transform = None

    if modeltype == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        load_checkpoint(model, model_path)
        # ResNet: 追踪最后一个卷积层的输出
        target_layers = [model.layer4[-1]]

    elif modeltype == "regnet":
        # 假设您用的是 regnet_y_3_2gf
        model = models.regnet_y_3_2gf(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        load_checkpoint(model, model_path)
        # RegNet: 追踪主干网络的最后一个 Block
        target_layers = [model.trunk_output[-1]]

    elif modeltype == "swin":
        model = timm.create_model(TIMM_SWIN_ID, pretrained=False, num_classes=num_classes)
        load_checkpoint(model, model_path)
        target_layers = [model.layers[-1].blocks[-1].norm1]
        reshape_transform = reshape_transform_swin

    elif modeltype == "swin_aligned":
        model = timm.create_model(TIMM_SWIN_ID, pretrained=False, num_classes=num_classes)
        load_checkpoint(model, model_path)
        target_layers = [model.layers[-1].blocks[-1].norm1]
        reshape_transform = reshape_transform_swin

    elif modeltype == "swin_diff":
        model = timm.create_model(TIMM_SWIN_ID, pretrained=False, num_classes=num_classes)
        model = inject_only_diff(model)
        load_checkpoint(model, model_path)
        target_layers = [model.layers[-1].blocks[-1].norm1]
        reshape_transform = reshape_transform_swin

    elif modeltype == "swin_dca_only":
        model = timm.create_model(TIMM_SWIN_ID, pretrained=False, num_classes=num_classes)
        model = inject_dynamic_residual_routing(model)
        load_checkpoint(model, model_path, normalize_fn=normalize_dca_checkpoint)
        target_layers = [model.layers[-1].blocks[-1].norm1]
        reshape_transform = reshape_transform_swin

    elif modeltype == "swin_diff_dca":
        model = get_swin_diff_dca_model(num_classes)
        load_checkpoint(model, model_path, normalize_fn=normalize_diff_dca_checkpoint)
        target_layers = [model.layers[-1].blocks[-1].norm1]
        reshape_transform = reshape_transform_swin

    elif modeltype == "vit":
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        load_checkpoint(model, model_path)
        target_layers = [model.encoder.layers[-1].ln_1]

        # ViT 专属特征重塑：去掉 CLS token，拼成 14x14
        def reshape_transform_vit(tensor, height=14, width=14):
            result = tensor[:, 1:, :].reshape(
                tensor.size(0), height, width, tensor.size(2)
            )
            return result.transpose(2, 3).transpose(1, 2)

        reshape_transform = reshape_transform_vit

    else:
        raise ValueError(f"❌ 不支持的模型类型: {modeltype}")

    model.to(DEVICE)
    model.eval()

    # 初始化 CAM
    cam = GradCAM(
        model=model, target_layers=target_layers, reshape_transform=reshape_transform
    )
    return model, cam


def main(modeltype, outputdir):
    # 1. 加载类别索引
    if not os.path.exists(CLASS_IDX_PATH):
        print(f"❌ 找不到类别索引文件: {CLASS_IDX_PATH}")
        return
    with open(CLASS_IDX_PATH, "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    # 2. 动态加载模型与 CAM
    try:
        model, cam = setup_model_and_cam(modeltype, num_classes)
    except FileNotFoundError:
        print(
            f"❌ 找不到 {modeltype} 的权重文件！请检查 CHECKPOINT_PATHS 里的路径是否正确。"
        )
        return

    # 3. 图像预处理流水线
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # 4. 自动扫描图片
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"}
    image_files = []
    for root, dirs, files in os.walk(IMAGE_DIR):
        for file in files:
            if os.path.splitext(file)[1] in valid_extensions:
                image_files.append(os.path.join(root, file))

    if not image_files:
        print(f"❌ 在 {IMAGE_DIR} 及其子文件夹下没有找到任何图片！")
        return

    print(
        f"🚀 共扫描到 {len(image_files)} 张图片，开始为 【{modeltype.upper()}】 生成批量热力图..."
    )

    correct_count = 0
    total_count = len(image_files)

    # 5. 开始生成
    for img_path in image_files:
        true_label = os.path.basename(os.path.dirname(img_path))
        filename = os.path.basename(img_path)
        name, ext = os.path.splitext(filename)

        pil_img = Image.open(img_path).convert("RGB")
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        img_for_vis = cv2.imdecode(
            np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        img_for_vis = cv2.resize(img_for_vis[:, :, ::-1], (224, 224))
        img_for_vis_float = np.float32(img_for_vis) / 255.0

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            pred_idx = probabilities.argmax().item()
            pred_class = idx_to_class[pred_idx]
            pred_score = probabilities[pred_idx].item()

        targets = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        visualization = show_cam_on_image(
            img_for_vis_float, grayscale_cam, use_rgb=True
        )
        heatmap_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)

        orig_bgr = cv2.cvtColor(img_for_vis, cv2.COLOR_RGB2BGR)
        combined_img = np.hstack((orig_bgr, heatmap_bgr))

        save_dir = os.path.join(outputdir, true_label)
        os.makedirs(save_dir, exist_ok=True)

        combined_filename = f"{name}_T({true_label})_P({pred_class}){ext}"
        combined_path = os.path.join(save_dir, combined_filename)
        cv2.imencode(ext, combined_img)[1].tofile(combined_path)

        if true_label == pred_class:
            correct_count += 1
            print(f"✅ [{true_label}/{filename}] -> 预测正确! 置信度: {pred_score:.1%}")
        else:
            print(
                f"❌ [{true_label}/{filename}] -> 预测错误! 错认成了: {pred_class} ({pred_score:.1%})"
            )

    accuracy = correct_count / total_count
    print("\n" + "=" * 55)
    print(f"🎉 【{modeltype.upper()}】 模型全部处理完成！")
    print(f"📊 该模型准确率: {correct_count}/{total_count} ({accuracy:.2%})")
    print(f"📁 分类热力图已保存在: {outputdir}")
    print("=" * 55)


if __name__ == "__main__":
    for model in sorted(MODEL_TYPE):
        outputdir = f"/nfs/spy/soybean_detect/data/gradCamImgT/output_{model}_all" #输出路径
        main(model,outputdir)
