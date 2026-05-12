import os
import sys
from collections import defaultdict

DETECTION_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../detection_yolo"))
if DETECTION_DIR not in sys.path:
    sys.path.insert(0, DETECTION_DIR)

from common import *

def find_duplicate_filenames(directory):
    file_map = defaultdict(list)

    # Walk through all subdirectories, filtering by those that start with "images-"
    for root, dirs, files in os.walk(directory):
        # Filter dirs to only include those that start with "images-"
        dirs[:] = [d for d in dirs if d.startswith("images-") or d.startswith("backgrounds")]
        
        for file in files:
            file_map[file].append(os.path.join(root, file))

    # Identify and print duplicate file names
    duplicates = {file: paths for file, paths in file_map.items() if len(paths) > 1}

    if duplicates:
        print("Duplicate file names found:")
        for filename, paths in duplicates.items():
            print(f"\nFile: {filename}")
            for path in paths:
                print(f" - {path}")
    else:
        print("No duplicate file names found.")

# Example Usage
directory_path = f'{base_src_dir}'  # Change this to your directory path
find_duplicate_filenames(directory_path)
