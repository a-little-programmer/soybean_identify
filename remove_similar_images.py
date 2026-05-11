import os
import shutil
import cv2
import numpy as np

from common import *

def dhash_cv2(image_path, hash_size):
    """Compute difference hash (dHash) of the image using OpenCV."""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (hash_size + 1, hash_size))
        diff = img[:, 1:] > img[:, :-1]
        return diff.flatten()
    except Exception:
        return None

def hamming_distance(hash1, hash2):
    return np.count_nonzero(hash1 != hash2)

def find_very_similar_images(directory, hash_size=128, hash_threshold=50):
    hashes = []
    paths = []
    review_dir = os.path.join(base_src_dir, 'to-review')
    os.makedirs(review_dir, exist_ok=True)
    counter = 0

    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.lower().endswith(image_extensions):
                fpath = os.path.join(root, fname)
                hash_vec = dhash_cv2(fpath, hash_size)
                if hash_vec is None:
                    continue

                found = False
                for i, h in enumerate(hashes):
                    if hamming_distance(hash_vec, h) <= hash_threshold:
                        print(f"{counter + 1}: Visually similar image\n  Original: {paths[i]}\n  Duplicate: {fpath}")
                        shutil.move(fpath, os.path.join(review_dir, fname))

                        shutil.copy(paths[i], review_dir)

                        counter += 1
                        found = True
                        break

                if not found:
                    hashes.append(hash_vec)
                    paths.append(fpath)


background_dir = os.path.join(base_src_dir, backgrounds_src_dir_name)

find_very_similar_images(base_src_dir, 128, 70)
find_very_similar_images(background_dir, 128, 500)