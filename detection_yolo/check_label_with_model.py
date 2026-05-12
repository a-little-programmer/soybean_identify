import json
import os
import random
import glob

from generate_label import *

def load_label(label_path):
    """Load a JSON label file."""
    with open(label_path, 'r') as f:
        label_data = json.load(f)
    return label_data

def extract_rectangles(label_data):
    """Extract rectangles from the LabelMe JSON format, ensuring the first point is top-left and the second is bottom-right."""
    rectangles = []
    for shape in label_data['shapes']:
        if shape['shape_type'] == 'rectangle':
            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][1]

            # Normalize points to ensure (x1, y1) is top-left and (x2, y2) is bottom-right
            x_min = min(x1, x2)
            y_min = min(y1, y2)
            x_max = max(x1, x2)
            y_max = max(y1, y2)

            rectangles.append([x_min, y_min, x_max, y_max])
    return rectangles

def calculate_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    # Compute the area of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both bounding boxes
    boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the IoU
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def compare_labels(generated_label_path, existing_label_path, iou_threshold=0.3):
    """Compare generated and existing labels to ensure rectangles match with IoU > 0.X and label classes match."""
    
    generated_label = load_label(generated_label_path)
    existing_label = load_label(existing_label_path)

    generated_rectangles = extract_rectangles(generated_label)
    existing_rectangles = extract_rectangles(existing_label)

    # Output total number of rectangles in each file
    print(f"Total rectangles in generated label file: {len(generated_rectangles)}")
    print(f"Total rectangles in existing label file: {len(existing_rectangles)}")

    # Check that every generated rectangle has a matching existing rectangle with the same label class
    for gen_rect, gen_label in zip(generated_rectangles, generated_label['shapes']):
        match_found = False
        max_iou = 0
        for ex_rect, ex_label in zip(existing_rectangles, existing_label['shapes']):
            iou = calculate_iou(gen_rect, ex_rect)

            # Check if the label class matches
            if gen_label['label'] != ex_label['label']:
                continue  # Skip this comparison if the labels are different

            if iou > max_iou:
                max_iou = iou
            if iou >= iou_threshold:
                match_found = True
                break
        if not match_found:
            print(f"No match found for generated rectangle {gen_rect} with label {gen_label['label']} in {generated_label_path} with IoU >= {iou_threshold}. Maximum IoU observed: {max_iou}")
            return False

    # Check that every existing rectangle has a matching generated rectangle with the same label class
    for ex_rect, ex_label in zip(existing_rectangles, existing_label['shapes']):
        match_found = False
        max_iou = 0
        for gen_rect, gen_label in zip(generated_rectangles, generated_label['shapes']):
            iou = calculate_iou(gen_rect, ex_rect)

            # Check if the label class matches
            if gen_label['label'] != ex_label['label']:
                continue  # Skip this comparison if the labels are different

            if iou > max_iou:
                max_iou = iou
            if iou >= iou_threshold:
                match_found = True
                break
        if not match_found:
            print(f"No match found for existing rectangle {ex_rect} with label {ex_label['label']} in {existing_label_path} with IoU >= {iou_threshold}. Maximum IoU observed: {max_iou}")
            return False

    return True

if __name__ == '__main__':
    # Assuming the directories are defined somewhere above
    # Get all label files from the source directory
    label_files = glob.glob(os.path.join(label_src_dir, f'*.json'))

    # Shuffle the list to ensure randomness
    random.shuffle(label_files)

    # Iterate over each label file
    for label_file in label_files:
        print("Checking ", label_file)
        # Construct the corresponding generated label path
        label_name = os.path.basename(label_file)
        generated_label_file = os.path.join(generated_label_src_dir, label_name)

        # Generated label exists, meaning the check has been done before
        if os.path.exists(generated_label_file):
            print(f'Expected label exists, {generated_label_file}')
            continue

        with open(label_file, 'r') as f:
            label_data = json.load(f)
        
        # Extract imagePath from the label JSON
        image_path = label_data['imagePath']

        # Ensure the image file exists
        image_file = os.path.join(label_src_dir, image_path)
        assert os.path.exists(image_file)

        # Generate the label
        result = generate_label(image_file, generated_label_src_dir_name)

        # Generate label failed, meaning the check failed
        if not result:
            print(f"CHECK FAIL: No detection for {label_name}")
            #exit()
            continue

        # Perform the comparison
        result = compare_labels(generated_label_file, label_file)

        # Output the result
        if result:
            print(f"CHECK OK: {label_name}")
        else:
            print(f"CHECK FAIL: {label_name}")
            exit()
