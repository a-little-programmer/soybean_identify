import os
import json
import cv2
from common import *

# Function to crop and save bounding boxes as individual images
def crop_and_save_bboxes(image_path, label_data, output_dir):
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over each bounding box in the label data
    for idx, shape in enumerate(label_data["shapes"]):
        # Extract bounding box coordinates (xmin, ymin, xmax, ymax)
        (xmin, ymin), (xmax, ymax) = shape["points"]
        
        # Ensure coordinates are within the image bounds and convert to integers
        xmin = max(0, min(int(xmin), img_width))
        xmax = max(0, min(int(xmax), img_width))
        ymin = max(0, min(int(ymin), img_height))
        ymax = max(0, min(int(ymax), img_height))

        # Calculate the width and height of the bounding box
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        
        # Skip bounding boxes with width or height < 100
        if bbox_width < 100 or bbox_height < 100:
            continue

        # Crop the image
        cropped_img = img[ymin:ymax, xmin:xmax]

        # Save the cropped image with a new filename
        output_filename = f"{label_data['imagePath'].split('/')[-1].split('.')[0]}-{idx + 1}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, cropped_img)
        print(f"Saved cropped image: {output_path}")


# Main function to process all JSON files and extract bounding boxes
def process_json_and_crop_bboxes_for_directory(label_src_dir, output_dir):
    # Loop through all JSON files in the provided directory
    for json_file in os.listdir(label_src_dir):
        if json_file.endswith(".json"):
            json_file_path = os.path.join(label_src_dir, json_file)
            with open(json_file_path, 'r') as f:
                label_data = json.load(f)
            
            # Get the image path
            image_path = os.path.join(os.path.dirname(json_file_path), label_data['imagePath'])
            
            # Call the function to crop and save bounding boxes
            crop_and_save_bboxes(image_path, label_data, output_dir)


# Create the "images-generated" folder inside the base directory if it doesn't exist
output_dir = os.path.join(base_src_dir, "images-generated")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process all label files and save cropped bounding boxes in the generated directory
process_json_and_crop_bboxes_for_directory(label_src_dir, output_dir)
