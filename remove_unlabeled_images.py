import os
import shutil

from common import *

# directories
images_dir = os.path.join(base_src_dir, "images")
label_dir = os.path.join(base_src_dir, label_src_dir_name)
output_dir = os.path.join(base_src_dir, 'unlabeled-images')

# create output dir if not exists
os.makedirs(output_dir, exist_ok=True)


# list all JSON files (without extension)
json_basenames = {
    os.path.splitext(fname)[0] 
    for fname in os.listdir(label_dir) 
    if fname.lower().endswith(".json")
}

moved_count = 0

for fname in os.listdir(images_dir):
    fpath = os.path.join(images_dir, fname)

    # skip directories
    if os.path.isdir(fpath):
        continue

    # check if it is an image
    ext = os.path.splitext(fname)[1].lower()
    if ext not in image_extensions:
        continue

    basename = os.path.splitext(fname)[0]

    # if no matching json → move
    if basename not in json_basenames:
        dst = os.path.join(output_dir, fname)
        print(f"Moving: {fname} → {dst}")
        shutil.move(fpath, dst)
        moved_count += 1

print(f"\nDone. Moved {moved_count} images without JSON labels.")
