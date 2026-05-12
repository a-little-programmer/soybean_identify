import os
import json

from common import *

# Directory containing your JSON label files
label_dir = os.path.join(base_src_dir, 'labels')

for filename in os.listdir(label_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(label_dir, filename)

        try:
            with open(file_path, 'r') as file:
                data = json.load(file)

            # Case 1: Empty label
            if "shapes" in data and len(data["shapes"]) == 0:
                print(f"Deleting empty label file: {file_path}")
                os.remove(file_path)
                continue

            # Case 2: Referenced image does not exist
            if "imagePath" in data:
                # Resolve relative path from label directory
                image_path = os.path.normpath(os.path.join(label_dir, data["imagePath"]))
                if not os.path.exists(image_path):
                    print(f"Deleting redundant label file (missing image): {file_path}")
                    os.remove(file_path)

        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {file_path}")
        except Exception as e:
            print(f"Unhandled error in file {file_path}: {e}")
