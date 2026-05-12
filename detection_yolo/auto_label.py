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
CONF_THRESHOLD = 0.25   # 置信度
IOU_THRESHOLD = 0.4     # NMS 阈值
EDGE_MARGIN = 50        # 边缘过滤距离
# -------------------------------------------

def generate_label_for_image(image_path, output_label_dir):
    file_name = os.path.basename(image_path)
    print(f"正在处理: {file_name} ...")
    
    # 1. 读取原始大图
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return False
        
    img_h, img_w = img.shape[:2]
    print(f"图片尺寸: {img_w}x{img_h}")

    # 2. 准备切片滑动窗口
    stride_h = int(SLICE_SIZE * (1 - OVERLAP_RATIO))
    stride_w = int(SLICE_SIZE * (1 - OVERLAP_RATIO))
    
    all_boxes = []   
    all_scores = []  
    all_classes = [] 
    
    # 3. 开始切片并检测
    for y in range(0, img_h, stride_h):
        for x in range(0, img_w, stride_w):
            x_min, y_min = x, y
            x_max = min(x + SLICE_SIZE, img_w)
            y_max = min(y + SLICE_SIZE, img_h)
            
            # 边缘修正
            if x_max - x_min < SLICE_SIZE and img_w >= SLICE_SIZE:
                x_min = img_w - SLICE_SIZE
            if y_max - y_min < SLICE_SIZE and img_h >= SLICE_SIZE:
                y_min = img_h - SLICE_SIZE
            
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
            
            # 边缘过滤
            for k in range(len(boxes_np)):
                box = boxes_np[k]
                bx1, by1, bx2, by2 = box
                
                is_touching_left = (bx1 < EDGE_MARGIN) and (x_min > 0)
                is_touching_top = (by1 < EDGE_MARGIN) and (y_min > 0)
                is_touching_right = (bx2 > SLICE_SIZE - EDGE_MARGIN) and (x_max < img_w)
                is_touching_bottom = (by2 > SLICE_SIZE - EDGE_MARGIN) and (y_max < img_h)
                
                if is_touching_left or is_touching_top or is_touching_right or is_touching_bottom:
                    continue 

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

    # 4. 全局 NMS (强制使用 float32 防止报错)
    boxes_tensor = torch.tensor(all_boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(all_scores, dtype=torch.float32)
    classes_tensor = torch.tensor(all_classes, dtype=torch.float32)
    
    keep_indices = nms(boxes_tensor, scores_tensor, IOU_THRESHOLD)
    
    final_boxes = boxes_tensor[keep_indices].numpy()
    final_classes = classes_tensor[keep_indices].numpy()
    final_scores = scores_tensor[keep_indices].numpy()
    
    print(f"  ✅ 检测到 {len(final_boxes)} 个目标。")

    # =========================================================
    # 🌟 修复重点区域：限制坐标范围 + 格式化输出 🌟
    # =========================================================
    
    # 【修复2】强制将坐标限制在图片范围内 (0 ~ img_w, 0 ~ img_h)
    # 防止因为模型预测偏离，导致坐标出现 -10 或 30000 这种情况
    np.clip(final_boxes[:, 0], 0, img_w, out=final_boxes[:, 0]) # x1
    np.clip(final_boxes[:, 1], 0, img_h, out=final_boxes[:, 1]) # y1
    np.clip(final_boxes[:, 2], 0, img_w, out=final_boxes[:, 2]) # x2
    np.clip(final_boxes[:, 3], 0, img_h, out=final_boxes[:, 3]) # y2

    # =========================================================
    # 🌟 新增部分：绘制并保存带框图片 🌟
    # =========================================================
    draw_img = img.copy()
    # 颜色定义 (B, G, R)
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)] 

    for i in range(len(final_boxes)):
        x1, y1, x2, y2 = map(int, final_boxes[i])
        cls_id = int(final_classes[i])
        score = final_scores[i]
        
        # 获取类别名
        if cls_id < len(label_names):
            class_name = label_names[cls_id]
        else:
            class_name = str(cls_id)

        color = colors[cls_id % len(colors)]
        
        # 画矩形框
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), color, 4)
        # 写类别和分数
        label_text = f"{class_name} {score:.2f}"
        cv2.putText(draw_img, label_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
    
    # 自动推导图片保存路径：
    # 如果 label 存在 .../generate/labels/
    # 图片就存在 .../generate/images/
    parent_dir = os.path.dirname(output_label_dir)
    output_img_dir = os.path.join(parent_dir, 'images')
    
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
        
    save_img_path = os.path.join(output_img_dir, file_name)
    cv2.imwrite(save_img_path, draw_img)
    print(f"  🖼️  效果图已保存: {save_img_path}")
    # =========================================================

# =========================================================
    # 🌟 修复核心：生成归一化 YOLO 数据 🌟
    # =========================================================
    yolo_lines = []
    
    for i in range(len(final_boxes)):
        x1, y1, x2, y2 = final_boxes[i]
        cls = int(final_classes[i])
        
        # 计算宽和高 (绝对像素)
        w_box = x2 - x1
        h_box = y2 - y1
        
        # 计算中心点 (绝对像素)
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        
        # 【修复3】添加 .6f 格式化，防止产生科学计数法
        # 必须确保 0 <= 结果 <= 1
        x_c_norm = x_center / img_w
        y_c_norm = y_center / img_h
        w_norm = w_box / img_w
        h_norm = h_box / img_h
        
        # 再次保险：限制在 0-1 之间
        x_c_norm = min(max(x_c_norm, 0.0), 1.0)
        y_c_norm = min(max(y_c_norm, 0.0), 1.0)
        w_norm = min(max(w_norm, 0.0), 1.0)
        h_norm = min(max(h_norm, 0.0), 1.0)

        # 格式化字符串，保留6位小数
        line = f'{cls} {x_c_norm:.6f} {y_c_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n'
        yolo_lines.append(line)

    # 保存 JSON
    root_name, _ = os.path.splitext(file_name)
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)
        
    label_path = os.path.join(output_label_dir, f'{root_name}.json')
    json_output = yolo_to_json_with_lines(yolo_lines, image_path, label_map)

    with open(label_path, 'w') as json_file:
        json_file.write(json_output)
        
    print(f"  📝 标签已保存: {label_path}\n")
    return True

if __name__ == '__main__':
    # --- 配置输入输出 ---
    # 注意：这里我们配置输出到 'generate' 文件夹，不直接覆盖 'labels'
    # 这样方便您检查，检查没问题了再手动复制到 labels 参与训练
    
    # 输入: 原始图片目录
    raw_images_dir = os.path.join(base_src_dir, 'backgrounds') 
    
    # 输出: 自动生成的标签存放目录
    # 会生成在 data/raw_data/generate/labels 和 data/raw_data/generate/images
    base_gen_dir = os.path.join(base_src_dir, 'generate')
    output_labels_dir = os.path.join(base_gen_dir, 'labels')
    
    print(f"输入目录: {raw_images_dir}")
    print(f"输出目录: {output_labels_dir}")
    
    extensions = ['*.jpg', '*.png', '*.JPG', '*.PNG']
    images_list = []
    for ext in extensions:
        images_list.extend(glob.glob(os.path.join(raw_images_dir, ext)))
    
    if not images_list:
        print(f"❌ 在 {raw_images_dir} 未找到图片。")
    else:
        print(f"📂 找到 {len(images_list)} 张图片，开始自动标注...")
        for image_path in images_list:
            generate_label_for_image(image_path, output_labels_dir)