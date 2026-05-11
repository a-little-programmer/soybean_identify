import random
import glob

from common import *
from generate_label import *


if __name__ == '__main__':
    list = glob.glob(f'{base_src_dir}/{image_src_dir_name}/*')

    # Shuffle the list in place
    random.shuffle(list)

    # Sort the list based on file names
    #sorted_list = sorted(list, key=lambda x: os.path.basename(x))

    label_count = 0  # Counter for the number of generate_label calls

    for image_path in list:
        if label_count >= 100:
            break  # Stop the loop after 10 calls to generate_label

        if label_exists_for_image(image_path):
            continue

        #boxes = detect(image_path)
        #if boxes is None or len(boxes) > 1:
        
        print(f'<<<<<<<<<<<<<<<<   {os.path.basename(image_path)}')
        generate_label(image_path)
        label_count += 1  # Increment the counter
