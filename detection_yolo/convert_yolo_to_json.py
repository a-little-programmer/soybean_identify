import os
import json
import cv2

from common import *

def yolo_to_json(yolo_label_file, image_path, label_map):
    with open(yolo_label_file, 'r') as f:
        yolo_lines = f.readlines()
    return yolo_to_json_with_lines(yolo_lines, image_path, label_map)

def yolo_to_json_with_lines(yolo_lines, image_path, label_map):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    image_height, image_width, _ = image.shape

    data = {
        "version": anylabeling_version,
        "flags": {},
        "shapes": [],
        "imagePath": f'../images/{os.path.basename(image_path)}',
        "imageData": None,
        "imageHeight": image_height,
        "imageWidth": image_width
    }
    
    for line in yolo_lines:
        label_id, x_center, y_center, width, height = map(float, line.split())
        
        x_center *= image_width
        y_center *= image_height
        width *= image_width
        height *= image_height
        
        x_min = x_center - (width / 2)
        y_min = y_center - (height / 2)
        x_max = x_center + (width / 2)
        y_max = y_center + (height / 2)
        
        shape = {
            "label": label_map[int(label_id)],
            "text": "",
            "points": [
                [x_min, y_min],
                [x_max, y_max]
            ],
            "group_id": None,
            "shape_type": "rectangle",
            "flags": {}
        }
        
        data["shapes"].append(shape)
    
    return json.dumps(data, indent = 2)

def process_all_files(labels_dir, images_dir, label_map):
    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            basename = os.path.splitext(label_file)[0]

            image_file = os.path.join(images_dir, f"{basename}.png")
            if not os.path.exists(image_file):
                image_file = os.path.join(images_dir, f"{basename}.jpg")
                if not os.path.exists(image_file):
                    print(f"Image file {image_file} does not exist for label {label_file}")
                    continue
            
            yolo_label_file = os.path.join(labels_dir, label_file)
            json_output = yolo_to_json(yolo_label_file, image_file, label_map)
            
            output_json_path = os.path.join(labels_dir, f"{basename}.json")
            with open(output_json_path, 'w') as out_file:
                out_file.write(json_output)
            print(f"Processed {label_file} and saved to {output_json_path}")


if __name__ == "__main__":
    labels_dir = os.path.join(base_src_dir, generated_label_src_dir_name)
    images_dir = os.path.join(base_src_dir, image_src_dir_name)

    process_all_files(labels_dir, images_dir, label_map)
