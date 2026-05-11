import os
import shutil
import cv2
import numpy as np
import hashlib

from common import *

def md5_hash(image_path):
    """Compute MD5 hash of an image file's binary content."""
    try:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

def find_identical_images(directory):
    hashes = {}
    review_dir = os.path.join(base_src_dir, 'to-review')
    os.makedirs(review_dir, exist_ok=True)
    counter = 0

    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.lower().endswith(image_extensions):
                fpath = os.path.join(root, fname)
                h = md5_hash(fpath)
                if h is None:
                    continue

                if h in hashes:
                    # This is a duplicate — move to "to-review"
                    print(f"{counter + 1}: Duplicate found — {fpath}")
                    shutil.move(fpath, os.path.join(review_dir, fname))

                    # Remove corresponding JSON
                    base_name = os.path.splitext(fname)[0]
                    json_path = os.path.join(base_src_dir, label_src_dir_name, base_name + '.json')
                    if os.path.exists(json_path):
                        os.remove(json_path)

                    counter += 1
                else:
                    hashes[h] = fpath

# Run the scan
find_identical_images(base_src_dir)
