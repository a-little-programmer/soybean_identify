import json
import os
import glob
import os.path as osp
import argparse

def convertJsonToYolo(json_file_path="", result_dir_path="", class_list=["dusty", "defect", "damaged"]):
    """
    This function converts LabelMe JSON annotations with rectangular shapes to YOLO format.
    :param json_file_path: Directory containing LabelMe JSON files.
    :param result_dir_path: Directory to save the converted YOLO TXT files.
    :param class_list: List of class labels.
    :return: None
    """
    # Create the result directory if it doesn't exist
    if not os.path.exists(result_dir_path):
        os.mkdir(result_dir_path)

    # Get all JSON files from the specified directory
    json_file_list = glob.glob(osp.join(json_file_path, "*.json"))

    # Iterate over each JSON file
    for json_file in json_file_list:
        # Open the JSON file
        with open(json_file, "r", encoding="utf-8") as f:
            file_in = json.load(f)

            # Extract shapes from the JSON file
            shapes = file_in.get("shapes", [])
            imageWidth = file_in.get("imageWidth")
            imageHeight = file_in.get("imageHeight")

            # Ensure image dimensions are present
            if imageWidth is None or imageHeight is None:
                print(f"Skipping {json_file}: Missing image dimensions.")
                continue

            # Open a corresponding TXT file for writing YOLO format annotations
            output_file_path = osp.join(result_dir_path, osp.basename(json_file).replace(".json", ".txt"))
            with open(output_file_path, "w") as file_handle:
                # Iterate over each shape
                for shape in shapes:
                    if shape["shape_type"] != "rectangle":
                        print(f"Skipping non-rectangle shape in {json_file}")
                        continue
                    
                    # Get the class index from the class list
                    class_index = class_list.index(shape["label"])

                    # Extract the two points
                    x1, y1 = shape["points"][0]
                    x2, y2 = shape["points"][1]

                    # Calculate the correct min/max coordinates
                    x_min = min(x1, x2)
                    y_min = min(y1, y2)
                    x_max = max(x1, x2)
                    y_max = max(y1, y2)

                    # Compute center, width, and height in normalized coordinates
                    x_center = (x_min + x_max) / 2.0 / imageWidth
                    y_center = (y_min + y_max) / 2.0 / imageHeight
                    width = (x_max - x_min) / imageWidth
                    height = (y_max - y_min) / imageHeight

                    # Write the class index and bounding box to the file
                    file_handle.write(f"{class_index} {x_center} {y_center} {width} {height}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LabelMe JSON annotations with rectangular shapes to YOLO format. Example:\
                                     python script_name.py /path/to/json/files /path/to/txt/files --class_list class1 class2 class3")
    parser.add_argument('src_path', type=str, help='The full path of the folder containing JSON files.')
    parser.add_argument('dst_path', type=str, help='The full path of the folder to save the TXT files.')
    parser.add_argument('--class_list', type=str, nargs='+', default=["dusty", "defect", "damaged"],
                        help='List of class names. Default: ["dusty", "defect", "damaged"]')
    
    args = parser.parse_args()
    
    convertJsonToYolo(json_file_path=args.src_path, result_dir_path=args.dst_path, class_list=args.class_list)
