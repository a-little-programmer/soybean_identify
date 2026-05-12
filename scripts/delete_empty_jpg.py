import os
import sys

DETECTION_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../detection_yolo"))
if DETECTION_DIR not in sys.path:
    sys.path.insert(0, DETECTION_DIR)

from common import *


# 删除没有对应json的图片
def delete_empty_jpg(base_dst_dir):
    images_dir = f'{base_dst_dir}/images-20241213'
    labels_dir = f'{base_dst_dir}/labels'

    for image_file in os.listdir(images_dir):
        if image_file.endswith('.jpg'):
            basename = os.path.splitext(image_file)[0]
            json_file = os.path.join(labels_dir, f"{basename}.json")
            if not os.path.exists(json_file):
                print(image_file, ':')
                # return
                os.remove(f'{images_dir}/{image_file}')
                print('删除成功')
                

if __name__ == "__main__":
    
    # source_dir = f"{base_src_dir}/{source_folder_name}"
    delete_empty_jpg(base_dst_dir)
