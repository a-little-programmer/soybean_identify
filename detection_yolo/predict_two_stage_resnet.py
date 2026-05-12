import torch
import cv2
import os
import json
import numpy as np
from PIL import Image
from torchvision import transforms, models
from ultralytics import YOLO
import torch.nn as nn
from tqdm import tqdm  # 引入进度条库，如果没有请 pip install tqdm

# ================= 配置区域 =================
# 1. 模型路径
YOLO_MODEL_PATH = "../data/target_data/soybean_detect.pt" 
RESNET_MODEL_PATH = "/nfs/spy/soybean_detect/code/result/checkpoints/best_resnet50_soybean.pth"
CLASS_INDICES_PATH = "/nfs/spy/soybean_detect/code/result/checkpoints/class_indices_resnet50.json"

# 2. 文件夹路径配置 (修改这里)
# 输入文件夹：存放待检测图片的目录
INPUT_DIR = "/nfs/spy/soybean_detect/data/raw_data/backgrounds" 

# 输出文件夹：存放处理结果的目录
OUTPUT_DIR = "/nfs/spy/soybean_detect/data/raw_data/generate/images"
# ===========================================

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_resnet(model_path, num_classes):
    print(f"Loading ResNet from {model_path}...")
    model = models.resnet50(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def process_single_image(img_path, save_path, yolo, resnet, idx_to_class, resnet_transforms):
    """处理单张图片的逻辑封装"""
    
    # 1. 读取图片
    # OpenCV 读取用于绘图
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        print(f"⚠️ 无法读取图片: {img_path}")
        return
        
    # PIL 读取用于 ResNet 裁剪 (转换颜色空间)
    try:
        pil_img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"⚠️ PIL 读取错误: {e}")
        return

    # 2. YOLO 检测
    # verbose=False 防止 YOLO 每一张图都打印很多日志刷屏
    results = yolo(img_path, verbose=False) 
    result = results[0]

    # 如果没检测到目标，直接保存原图并返回
    if len(result.boxes) == 0:
        cv2.imwrite(save_path, orig_img)
        return

    # 3. 遍历每个目标进行 ResNet 修正
    for box in result.boxes:
        # --- 获取 YOLO 结果 ---
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        yolo_conf = float(box.conf)
        yolo_cls_id = int(box.cls)
        yolo_label = result.names[yolo_cls_id]

        # --- 裁剪 (Crop) ---
        pad = 5 # 稍微外扩一点
        crop = pil_img.crop((
            max(0, x1-pad), 
            max(0, y1-pad), 
            min(pil_img.width, x2+pad), 
            min(pil_img.height, y2+pad)
        ))
        
        # --- ResNet 分类 ---
        input_tensor = resnet_transforms(crop).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = resnet(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            
        resnet_conf = conf.item()
        resnet_label = idx_to_class[pred_idx.item()]

        # --- 决策逻辑 ---
        # 这里使用了完全信任 ResNet 的策略
        final_label = resnet_label 
        final_conf = resnet_conf

        # --- 绘图 ---
        color = (0, 255, 0) # 绿色 (一致)
        if yolo_label != resnet_label:
            color = (0, 0, 255) # 红色 (发生修正)
            
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), color, 2)
        
        # 标签格式: 类别 置信度
        text = f"{final_label} {final_conf:.2f}"
        
        # 防止文字写到图片外面
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.putText(orig_img, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 4. 保存结果
    cv2.imwrite(save_path, orig_img)

def main():
    # 1. 检查输入目录
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 输入目录不存在: {INPUT_DIR}")
        return

    # 2. 创建输出目录 (如果不存在自动创建)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📂 已创建输出目录: {OUTPUT_DIR}")

    # 3. 加载类别映射
    if not os.path.exists(CLASS_INDICES_PATH):
        print("❌ 找不到 class_indices.json")
        return
    
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    # 4. 加载模型 (只加载一次，不要在循环里加载)
    print("⏳ 正在加载模型...")
    resnet = load_resnet(RESNET_MODEL_PATH, num_classes)
    yolo = YOLO(YOLO_MODEL_PATH)
    
    resnet_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 5. 获取所有图片文件
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.JPG', '.PNG')
    image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(valid_exts)]
    
    if len(image_files) == 0:
        print(f"⚠️ 输入目录 {INPUT_DIR} 为空或没有图片文件。")
        return

    print(f"\n🚀 开始批量处理: 共 {len(image_files)} 张图片")
    print(f"📂 输入: {INPUT_DIR}")
    print(f"📂 输出: {OUTPUT_DIR}")

    # 6. 循环处理
    # tqdm 用于显示进度条
    for img_name in tqdm(image_files, desc="Processing"):
        input_path = os.path.join(INPUT_DIR, img_name)
        output_path = os.path.join(OUTPUT_DIR, img_name)
        
        process_single_image(
            input_path, 
            output_path, 
            yolo, 
            resnet, 
            idx_to_class, 
            resnet_transforms
        )

    print(f"\n✅ 所有图片处理完成！结果保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
