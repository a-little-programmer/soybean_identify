import os
import json
import cv2
import glob
from tqdm import tqdm

# 默认配置（如果直接运行此脚本时使用）
DEFAULT_INPUT_IMG_DIR = "../data/raw_data/images"
DEFAULT_INPUT_LBL_DIR = "../data/raw_data/labels"
DEFAULT_OUTPUT_DIR = "../data/sliced_data"

def setup_dirs(output_image_dir, output_label_dir):
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

def compute_iou(box1, box2):
    # box: [xmin, ymin, xmax, ymax]
    xx1 = max(box1[0], box2[0])
    yy1 = max(box1[1], box2[1])
    xx2 = min(box1[2], box2[2])
    yy2 = min(box1[3], box2[3])

    w = max(0, xx2 - xx1)
    h = max(0, yy2 - yy1)
    inter = w * h
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    return inter / (area1 + 1e-6)

def slice_single_image(image_path, label_path, output_img_dir, output_lbl_dir, slice_h, slice_w, overlap, iou_thresh, min_area):
    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        return
    
    img_h, img_w = img.shape[:2]
    filename = os.path.basename(image_path)
    basename, ext = os.path.splitext(filename)

    # 读取标签
    shapes = []
    if os.path.exists(label_path):
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
                shapes = label_data.get('shapes', [])
        except Exception as e:
            print(f"Error reading JSON {label_path}: {e}")
            return

    # 计算滑动窗口的步长
    stride_h = int(slice_h * (1 - overlap))
    stride_w = int(slice_w * (1 - overlap))

    idx = 0
    # 开始滑动窗口
    for y in range(0, img_h, stride_h):
        for x in range(0, img_w, stride_w):
            x_min, y_min = x, y
            x_max = min(x + slice_w, img_w)
            y_max = min(y + slice_h, img_h)
            
            # 边缘修正
            if x_max - x_min < slice_w and img_w >= slice_w:
                x_min = img_w - slice_w
            if y_max - y_min < slice_h and img_h >= slice_h:
                y_min = img_h - slice_h
            
            x_max = x_min + slice_w
            y_max = y_min + slice_h
            
            crop_img = img[y_min:y_max, x_min:x_max]
            
            slice_name = f"{basename}_slice_{idx}"
            save_img_path = os.path.join(output_img_dir, f"{slice_name}{ext}")
            save_json_path = os.path.join(output_lbl_dir, f"{slice_name}.json")

            new_shapes = []
            slice_box = [x_min, y_min, x_max, y_max]

            for shape in shapes:
                if shape['shape_type'] != 'rectangle':
                    continue
                
                pts = shape['points']
                obj_xmin = min(pts[0][0], pts[1][0])
                obj_ymin = min(pts[0][1], pts[1][1])
                obj_xmax = max(pts[0][0], pts[1][0])
                obj_ymax = max(pts[0][1], pts[1][1])
                obj_box = [obj_xmin, obj_ymin, obj_xmax, obj_ymax]

                overlap_rate = compute_iou(obj_box, slice_box)
                if overlap_rate < iou_thresh:
                    continue

                new_xmin = max(0, obj_xmin - x_min)
                new_ymin = max(0, obj_ymin - y_min)
                new_xmax = min(slice_w, obj_xmax - x_min)
                new_ymax = min(slice_h, obj_ymax - y_min)

                if (new_xmax - new_xmin) * (new_ymax - new_ymin) < min_area:
                    continue

                new_shape = shape.copy()
                new_shape['points'] = [[new_xmin, new_ymin], [new_xmax, new_ymax]]
                new_shapes.append(new_shape)

            # 保存切片
            if len(new_shapes) > 0: # 仅保留有目标的切片
                cv2.imwrite(save_img_path, crop_img)
                json_content = {
                    "version": label_data.get("version", "0.4.30"),
                    "flags": {},
                    "shapes": new_shapes,
                    "imagePath": f"../images/{slice_name}{ext}",
                    "imageData": None,
                    "imageHeight": crop_img.shape[0],
                    "imageWidth": crop_img.shape[1]
                }
                with open(save_json_path, 'w') as f:
                    json.dump(json_content, f, indent=2)
            
            idx += 1

# =======================================================
#  对外暴露的接口函数
# =======================================================
def run_slicing(input_img_dir, input_label_dir, output_base_dir, 
                slice_size=2048, overlap=0.2):
    """
    主调函数：被 prepare_train.py 调用
    """
    print(f"\n[Slice] Starting image slicing...")
    print(f"[Slice] Input: {input_img_dir}")
    print(f"[Slice] Output: {output_base_dir}")
    
    out_img_dir = os.path.join(output_base_dir, "images")
    out_lbl_dir = os.path.join(output_base_dir, "labels")
    setup_dirs(out_img_dir, out_lbl_dir)

    # 扫描图片
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_img_dir, ext)))
    
    print(f"[Slice] Found {len(image_files)} original images.")
    
    # 开始处理
    for img_path in tqdm(image_files, desc="Slicing"):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(input_label_dir, f"{basename}.json")
        
        if not os.path.exists(json_path):
            # 尝试在同级目录找
            json_path = os.path.splitext(img_path)[0] + ".json"
        
        slice_single_image(img_path, json_path, out_img_dir, out_lbl_dir, 
                           slice_size, slice_size, overlap, 0.3, 100)
    
    print(f"[Slice] Slicing complete. Data saved to {output_base_dir}\n")

if __name__ == "__main__":
    # 允许单独运行测试
    run_slicing(DEFAULT_INPUT_IMG_DIR, DEFAULT_INPUT_LBL_DIR, DEFAULT_OUTPUT_DIR)