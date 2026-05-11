import torch
import cv2
import os
import glob
import numpy as np
from torchvision.ops import nms

from common import *
from convert_yolo_to_json import yolo_to_json_with_lines

# ----------------- 配置参数 -----------------
SLICE_SIZE = 2048       # 切片大小
OVERLAP_RATIO = 0.2     # 重叠率
CONF_THRESHOLD = 0.25   # 置信度 (切片后目标变大了，可以适当提高)
IOU_THRESHOLD = 0.4     # NMS 阈值
EDGE_MARGIN = 50        # [新增] 边缘过滤距离(像素)，防止切片边缘出现重影框
# -------------------------------------------

def generate_label_with_slicing(image_path):
    file_name = os.path.basename(image_path)
    print(f"正在处理: {file_name} ...")
    
    # 1. 动态确定保存路径
    # 假设输入是: .../data/raw_data/backgrounds/IMG.png
    # 我们希望输出到: .../data/raw_data/generate/images/IMG.png
    
    parent_dir = os.path.dirname(image_path)          # .../backgrounds
    grandparent_dir = os.path.dirname(parent_dir)      # .../raw_data
    
    # 定义 generate 根目录 (在 raw_data 下面创建 generate 文件夹)
    base_generate_dir = os.path.join(grandparent_dir, 'generate')
    
    detect_save_dir = os.path.join(base_generate_dir, 'images')
    label_save_dir = os.path.join(base_generate_dir, 'labels')
    
    # 自动创建目录
    os.makedirs(detect_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)

    # 2. 读取原始大图
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return False
        
    img_h, img_w = img.shape[:2]
    
    # 3. 准备切片滑动窗口
    stride_h = int(SLICE_SIZE * (1 - OVERLAP_RATIO))
    stride_w = int(SLICE_SIZE * (1 - OVERLAP_RATIO))
    
    all_boxes = []   
    all_scores = []  
    all_classes = [] 
    
    # 4. 开始切片并检测
    for y in range(0, img_h, stride_h):
        for x in range(0, img_w, stride_w):
            # 计算切片坐标
            x_min, y_min = x, y
            x_max = min(x + SLICE_SIZE, img_w)
            y_max = min(y + SLICE_SIZE, img_h)
            
            # 边缘修正：保证切片大小一致（除非图本身就很小）
            if x_max - x_min < SLICE_SIZE and img_w >= SLICE_SIZE:
                x_min = img_w - SLICE_SIZE
            if y_max - y_min < SLICE_SIZE and img_h >= SLICE_SIZE:
                y_min = img_h - SLICE_SIZE
            
            # 更新修正后的坐标
            x_max = x_min + SLICE_SIZE
            y_max = y_min + SLICE_SIZE
                
            slice_img = img[y_min:y_max, x_min:x_max]
            
            # 模型预测
            results = detect_model.predict(
                source=slice_img,
                device=device_,
                imgsz=model_image_size, 
                conf=CONF_THRESHOLD,
                iou=0.6,
                verbose=False 
            )
            
            if results[0].boxes is None:
                continue
                
            boxes_np = results[0].boxes.xyxy.cpu().numpy()
            scores_np = results[0].boxes.conf.cpu().numpy()
            classes_np = results[0].boxes.cls.cpu().numpy()
            
            # --- 🌟 核心逻辑：边缘过滤 (Edge Filtering) 🌟 ---
            # 如果检测框紧贴着切片的内部边缘，说明它被切断了，大概率是残次品。
            # 我们直接丢弃它，相信相邻的切片会在中心位置完整检测到它。
            for k in range(len(boxes_np)):
                box = boxes_np[k]
                bx1, by1, bx2, by2 = box
                
                # 检查是否贴边 (且不是原图的真实边缘)
                is_touching_left = (bx1 < EDGE_MARGIN) and (x_min > 0)
                is_touching_top = (by1 < EDGE_MARGIN) and (y_min > 0)
                is_touching_right = (bx2 > SLICE_SIZE - EDGE_MARGIN) and (x_max < img_w)
                is_touching_bottom = (by2 > SLICE_SIZE - EDGE_MARGIN) and (y_max < img_h)
                
                if is_touching_left or is_touching_top or is_touching_right or is_touching_bottom:
                    continue # 丢弃，解决重影问题

                # 坐标换算：小图 -> 大图
                global_box = [
                    box[0] + x_min, 
                    box[1] + y_min, 
                    box[2] + x_min, 
                    box[3] + y_min  
                ]
                all_boxes.append(global_box)
                all_scores.append(scores_np[k])
                all_classes.append(classes_np[k])

    if not all_boxes:
        print("  ⚠️ 未检测到任何目标。")
        return False

    # 5. 全局 NMS (最后一道防线)
    boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
    classes_tensor = torch.tensor(all_classes, dtype=torch.float32)
    
    keep_indices = nms(boxes_tensor, scores_tensor, IOU_THRESHOLD)
    
    final_boxes = boxes_tensor[keep_indices].numpy()
    final_classes = classes_tensor[keep_indices].numpy()
    final_scores = scores_tensor[keep_indices].numpy()
    
    print(f"  ✅ 合并后最终检测到 {len(final_boxes)} 个目标。")

    # 6. 绘制并保存结果图
    draw_img = img.copy()
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)] 

    for i in range(len(final_boxes)):
        x1, y1, x2, y2 = map(int, final_boxes[i])
        cls_id = int(final_classes[i])
        score = final_scores[i]
        
        class_name = label_names[cls_id] if cls_id < len(label_names) else str(cls_id)
        color = colors[cls_id % len(colors)]
        
        # 画框
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 4)
        # 写字
        label_text = f"{class_name} {score:.2f}"
        cv2.putText(draw_img, label_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)

    save_img_path = os.path.join(detect_save_dir, file_name)
    cv2.imwrite(save_img_path, draw_img)
    print(f"  🖼️  结果图已保存: {save_img_path}")

    # 7. 生成并保存 JSON 标签
    yolo_lines = []
    
    for i in range(len(final_boxes)):
        x1, y1, x2, y2 = final_boxes[i]
        cls = int(final_classes[i])
        
        # 转换为 YOLO 格式 (x_center, y_center, w, h) 归一化
        w_box = x2 - x1
        h_box = y2 - y1
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        
        line = f'{cls} {x_center / img_w} {y_center / img_h} {w_box / img_w} {h_box / img_h}\n'
        yolo_lines.append(line)

    root_name, _ = os.path.splitext(file_name)
    label_path = os.path.join(label_save_dir, f'{root_name}.json')

    json_output = yolo_to_json_with_lines(yolo_lines, image_path, label_map)

    with open(label_path, 'w') as json_file:
        json_file.write(json_output)
        
    print(f"  📝 标签已保存: {label_path}\n")
    return True

if __name__ == '__main__':
    # ---------------- 配置输入目录 ----------------
    # 填写您存放原始 6000x4000 图片的目录
    raw_images_dir = '/nfs/spy/soybean_detect/data/raw_data/backgrounds' 
    # --------------------------------------------
    
    extensions = ['*.jpg', '*.png', '*.JPG', '*.PNG']
    images_list = []
    for ext in extensions:
        images_list.extend(glob.glob(os.path.join(raw_images_dir, ext)))
    
    if not images_list:
        print(f"❌ 在 {raw_images_dir} 未找到图片，请检查路径是否正确。")
    else:
        print(f"📂 找到 {len(images_list)} 张图片，开始处理...\n")
        for image_path in images_list:
            generate_label_with_slicing(image_path)