import torch
import glob
import os
import datetime

from common import *
from convert_yolo_to_json import yolo_to_json_with_lines

# Return True for successful generation, False for no generation
def generate_label(image_path, label_output=label_src_dir_name):
    # Get the current date and time
    current_datetime = datetime.datetime.now()
    # Format it as a string
    current_datetime_string = current_datetime.strftime("%Y-%m-%d-%H_%M_%S")

    file_name = os.path.basename(image_path)

    # /Users/hjx/Documents/research/data-label/labware
    parent_directory = os.path.dirname(os.path.dirname(image_path))
    
    root, extension = os.path.splitext(file_name)

    label_path = os.path.join(parent_directory, label_output, f'{root}.json')
    if os.path.exists(label_path):
        print(f'Label exists, {label_path}')
        return False

    img_numpy = get_image_numpy(image_path)
    save_dir = '/nfs/spy/soybean_detect/data/raw_data'
    results = detect_model.predict(
        source=image_path, 
        device=device_,
        imgsz=model_image_size, 
        max_det=yolo_max_det,
        iou=0.5, 
        conf=0.15,
        #agnostic_nms=True,   #class-agnostic NMS, default is false
        save=True, 
        save_txt=True,
        save_conf=True,
        project=save_dir,
        name='generate',
        exist_ok=True,
        line_width=1
    )

    if results[0].boxes is None:
        return False
    
    boxes = results[0].boxes.xywh.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    
    if boxes.shape[0] == 0:
        return False
    
    height = img_numpy.shape[0]
    width = img_numpy.shape[1]

    # Prepare YOLO lines
    yolo_lines = []
    for i in range(boxes.shape[0]):
        (x, y, w, h) = boxes[i]
        if w == 0 or h == 0:
            continue
        yolo_lines.append(f'{int(classes[i])} {x / width} {y / height} {w / width} {h / height}\n')

    # Use yolo_to_json_with_lines to convert to JSON
    json_output = yolo_to_json_with_lines(yolo_lines, image_path, label_map)

    # Save the JSON output
    with open(label_path, 'w') as json_file:
        json_file.write(json_output)
    
    return True

# if __name__ == '__main__':
#     images_list = glob.glob(sample_image_path)
#     for image_path in images_list:
#         result = generate_label(image_path)
#         print(f"Processed {image_path}: Success = {result}")
if __name__ == '__main__':
    # 原始代码：只匹配一个文件
    # images_list = glob.glob(sample_image_path)
    
    # 更改为：匹配 backgrounds 目录下的所有 .png 和 .jpg 文件
    # 注意：这里假设 glob 和 os 库已导入 (generate_label.py 中已导入 glob 和 os)
    image_dir = os.path.join(base_src_dir, image_src_dir_name)
    images_list = glob.glob(os.path.join(image_dir, '*.png')) + \
                  glob.glob(os.path.join(image_dir, '*.jpg'))
    
    if not images_list:
        print(f"Warning: No images found in {image_dir}")
    
    for image_path in images_list:
        result = generate_label(image_path)
        print(f"Processed {image_path}: Success = {result}")