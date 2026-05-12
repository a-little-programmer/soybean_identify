# -*- coding: utf-8 -*-
import os
import json
import cv2
import glob
from tqdm import tqdm
import shutil
import numpy as np
import random

# ================= 配置区域 =================
# 1. 输入路径
YOLO_IMAGES_DIR = "../../data/raw_data/images"
YOLO_LABELS_DIR = "../../data/raw_data/labels"

# 2. 输出路径
OUTPUT_CLASSIFIER_DIR = "../../data/classifier_dataset_hsv"
# 保留中间裁剪结果：只按标注框裁出豆粒，不做黑边正方形填充，也不做 HSV 增强。
OUTPUT_RAW_CROP_DIR = "../../data/classifier_dataset_hsv_raw_crop"

# 3. 类别列表
CLASS_NAMES = [
    'nn49', 'nn60', 'zld105', 'sn29', 'lk314',
    'nn47', 'jd17', 'sd30', 'hd16', 'jng20839',
    'nn43', 'nn42', 'sn23', 'b73', 'sz2',
    'zd57', 'xd18', 'zd53', 'zd61', 'zd59',
    'zd51', 'sd29', 'xzd1', 'zh301', 'nn55',
]

# 4. 裁剪与切分参数
MIN_CROP_SIZE = 0
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1
SEED = 42

# 四周强制留出的黑边比例 (0.05 表示四周各留出最大边长 5% 的安全边距)
MARGIN_RATIO = 0.1

# 5. 数据增强配置 (仅对训练集生效)
ENABLE_AUGMENT = True

AUGMENT_PER_IMAGE = 2
AUG_INTENSITY = 0.3

# ===========================================

def augment_color(img, intensity=0.3):
    """色彩增强逻辑"""
    img = img.astype(np.float32)
    alpha = 1.0 + random.uniform(-intensity, intensity)
    beta = random.uniform(-intensity * 100, intensity * 100)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    s = s * (1.0 + random.uniform(-intensity, intensity))
    s = np.clip(s, 0, 255)
    h = h + random.uniform(-intensity * 10, intensity * 10)
    h[h < 0] += 180
    h[h > 179] -= 180

    hsv_aug = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_aug.astype(np.uint8), cv2.COLOR_HSV2BGR)

# 带安全边距的黑边正方形填充
def pad_to_square_with_margin(img, margin_ratio=MARGIN_RATIO, pad_color=(0, 0, 0)):
    h, w = img.shape[:2]
    max_dim = max(h, w)

    # 计算强制添加的黑边厚度
    margin = int(max_dim * margin_ratio)

    # 目标正方形的最终边长 = 原本的最大边长 + 两侧的黑边缓冲
    target_dim = max_dim + 2 * margin

    # 计算上下左右最终需要填充的像素量，使其绝对居中
    top = (target_dim - h) // 2
    bottom = target_dim - h - top
    left = (target_dim - w) // 2
    right = target_dim - w - left

    # 使用 cv2.copyMakeBorder 填充黑边
    square_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    return square_img

def setup_output_dirs(output_base_dir, class_names):
    """创建 output/train, output/val, output/test 目录"""
    if os.path.exists(output_base_dir):
        print(f"清理旧目录: {output_base_dir}")
        try:
            shutil.rmtree(output_base_dir)
        except Exception:
            pass

    for subset in ['train', 'val', 'test']:
        for class_name in class_names:
            os.makedirs(os.path.join(output_base_dir, subset, class_name), exist_ok=True)

def process_single_image(image_path, label_path, output_base_dir, raw_crop_base_dir, class_names, subset):
    """
    处理单张大图：读取 -> 裁剪 -> (增强) -> 保存到对应 subset 文件夹
    """
    # cv2.imdecode 可以处理中文路径，比 cv2.imread 更稳
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None: return 0

    basename = os.path.splitext(os.path.basename(image_path))[0]

    if not os.path.exists(label_path): return 0
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            shapes = json.load(f).get('shapes', [])
    except: return 0

    count = 0
    subset_dir = os.path.join(output_base_dir, subset)

    for i, shape in enumerate(shapes):
        if shape['label'] not in class_names: continue
        pts = shape['points']
        try:
            x1, y1 = int(min(pts[0][0], pts[1][0])), int(min(pts[0][1], pts[1][1]))
            x2, y2 = int(max(pts[0][0], pts[1][0])), int(max(pts[0][1], pts[1][1]))
        except: continue

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

        if (x2-x1) < MIN_CROP_SIZE or (y2-y1) < MIN_CROP_SIZE: continue

        crop_img = img[y1:y2, x1:x2]
        if crop_img.size == 0: continue

        save_dir = os.path.join(subset_dir, shape['label'])
        raw_save_dir = os.path.join(raw_crop_base_dir, subset, shape['label'])

        # 1. 保存中间裁剪图：未填黑边、未做训练增强
        raw_save_path = os.path.join(raw_save_dir, f"{basename}_{i}.jpg")
        cv2.imencode('.jpg', crop_img)[1].tofile(raw_save_path)

        # 2. 保存分类训练图：裁剪后填充黑边为正方形
        # 使用 cv2.imencode 支持中文路径写入
        crop_img_padded = pad_to_square_with_margin(crop_img, margin_ratio=MARGIN_RATIO, pad_color=(0, 0, 0))
        save_path = os.path.join(save_dir, f"{basename}_{i}.jpg")
        cv2.imencode('.jpg', crop_img_padded)[1].tofile(save_path)
        count += 1

        # 3. 训练集增强：先增强裁剪区域，再填充黑边，保持 padding 区域纯黑且分布一致
        if subset == 'train' and ENABLE_AUGMENT:
            for k in range(AUGMENT_PER_IMAGE):
                aug_img = augment_color(crop_img, AUG_INTENSITY)
                aug_img = pad_to_square_with_margin(aug_img, margin_ratio=MARGIN_RATIO, pad_color=(0, 0, 0))
                aug_save_path = os.path.join(save_dir, f"{basename}_{i}_aug{k}.jpg")
                cv2.imencode('.jpg', aug_img)[1].tofile(aug_save_path)
                count += 1
    return count

def main():
    random.seed(SEED)
    setup_output_dirs(OUTPUT_CLASSIFIER_DIR, CLASS_NAMES)
    setup_output_dirs(OUTPUT_RAW_CROP_DIR, CLASS_NAMES)

    print("开始扫描图片文件...")

    # === 支持多种格式 ===
    # 支持的扩展名列表 (不区分大小写)
    valid_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.PNG', '*.BMP']
    image_files = []

    for ext in valid_extensions:
        # glob 查找
        found = glob.glob(os.path.join(YOLO_IMAGES_DIR, ext))
        image_files.extend(found)

    # 去重（防止部分系统下 *.jpg 和 *.JPG 重复匹配）并排序
    image_files = sorted(list(set(image_files)))

    if not image_files:
        print(f"错误: 在 {YOLO_IMAGES_DIR} 下没有找到任何图片，请检查路径。")
        return

    # 2. 按“大图”进行打乱和切分
    random.shuffle(image_files)
    total = len(image_files)
    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * (TRAIN_RATIO + VAL_RATIO))

    splits = {
        'train': image_files[:n_train],
        'val':   image_files[n_train:n_val],
        'test':  image_files[n_val:]
    }

    print(f"数据集划分 (按大图): Total={total}")
    print(f"   Train: {len(splits['train'])} 张大图")
    print(f"   Val  : {len(splits['val'])} 张大图")
    print(f"   Test : {len(splits['test'])} 张大图")

    # 3. 执行裁剪
    for subset, files in splits.items():
        total_crops = 0
        print(f"\n正在处理 {subset} 集...")
        for img_path in tqdm(files):
            basename = os.path.splitext(os.path.basename(img_path))[0]
            # 假设 json 文件名和图片名一致
            json_path = os.path.join(YOLO_LABELS_DIR, f"{basename}.json")

            total_crops += process_single_image(
                img_path,
                json_path,
                OUTPUT_CLASSIFIER_DIR,
                OUTPUT_RAW_CROP_DIR,
                CLASS_NAMES,
                subset
            )
        print(f"   -> {subset} 集处理完毕，共生成 {total_crops} 张豆子切片")

if __name__ == "__main__":
    main()
