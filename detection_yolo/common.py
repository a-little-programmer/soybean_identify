import os
import cv2
import numpy as np
import torch

from ultralytics import YOLO

device_ = torch.device('cuda')

model_version = '1.0.0'

def get_model_version():
    return model_version


anylabeling_version = '0.4.30'

image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

# Whether to prepare training using gray scale images
use_gray_scale = False

base_code_dir = '/nfs/spy/soybean_detect/'
base_src_dir = '/nfs/spy/soybean_detect/data/raw_data'
base_slice_dir = '/nfs/spy/soybean_detect/data/sliced_data'
base_dst_dir = '/nfs/spy/soybean_detect/data/target_data'
base_generate_dir = '/nfs/spy/soybean_detect/data/raw_data/generate/'

image_src_dir_name = 'backgrounds'
backgrounds_src_dir_name = 'backgrounds'

label_src_dir_name = 'labels'
label_src_dir = os.path.join(base_src_dir, label_src_dir_name)
generated_label_src_dir_name = 'labels-generated'
generated_label_src_dir = os.path.join(base_src_dir, generated_label_src_dir_name)

convert_script = os.path.join(base_code_dir, 'code/convert_json_to_yolo.py')

label_names = ['nn49','nn60','zld105','sn29','lk314','nn47','jd17','hd16','jng20839','nn43','nn42','sn23','b73','sz2','zd57','xd18','zd53','zd61','zd59','zd51','sd30','sd29','xzd1','zh301','nn55']

label_map = {
    0: "nn49",
    1: "nn60",
    2: "zld105",
    3: "sn29",
    4: "lk314",
    5: "nn47",
    6: "jd17",
    7: "hd16",
    8: "jng20839",
    9: "nn43",
    10: "nn42",
    11: "sn23",
    12: "b73",
    13: "sz2",
    14: "zd57",
    15: "xd18",
    16: "zd53",
    17: "zd61",
    18: "zd59",
    19: "zd51",
    20: "sd30",
    21: "sd29",
    22: "xzd1",
    23: "zh301",
    24: "nn55"
}

model_image_size = 1024

yolo_max_det = 300

model_path = f'{base_dst_dir}/soybean_detect.pt'

if os.path.exists(model_path):
    detect_model = YOLO(model_path)


def get_image_numpy(image_path):
    if use_gray_scale:
        img_numpy = cv2.cvtColor(get_gray_numpy(image_path), cv2.COLOR_GRAY2BGR)
    else:
        img_numpy = cv2.imread(image_path)
    return img_numpy


def get_gray_numpy(image_path):
    img_numpy = cv2.imread(image_path)
    #start_time = time.time()
    #print(f"Your function took {time.time() - start_time:.6f} seconds to run.")
    return cv2.cvtColor(img_numpy, cv2.COLOR_BGR2GRAY)


def convert_to_grayscale(image_path, output_dir):
    gray_image = get_gray_numpy(image_path)
    file_name = os.path.basename(image_path)
    output_image_path = os.path.join(output_dir, file_name)
    cv2.imwrite(output_image_path, gray_image)


def convert_to_required_size(image_path, output_dir, convert_to_gray=use_gray_scale):
    """
    Converts the image to grayscale or keeps it in RGB, resizes it to the required size 
    while maintaining the aspect ratio, and saves it to the specified output directory.
    
    :param image_path: The path to the input image.
    :param output_dir: The directory where the resized image will be saved.
    :param convert_to_gray: If True, converts the image to grayscale before resizing. 
                            If False, keeps the image in RGB.
    """
    # Read the original image
    image = get_gray_numpy(image_path) if convert_to_gray else cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Unable to load image at {image_path}")

    # Get the original dimensions
    original_height, original_width = image.shape[:2]

    # Calculate the scaling factor
    scaling_factor = model_image_size / max(original_width, original_height)

    # Calculate the new dimensions
    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)

    # Resize the image while maintaining the aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Prepare the output path
    file_name = os.path.basename(image_path)
    output_image_path = os.path.join(output_dir, file_name)

    # Save the resized image
    cv2.imwrite(output_image_path, resized_image)
    #print(f"Resized {'grayscale' if convert_to_gray else 'RGB'} image saved at {output_image_path}")


# Check label existence for a image file
def label_exists_for_image(image_path):
    file_name = os.path.basename(image_path)
    parent_directory = os.path.dirname(image_path)
    parent_directory = os.path.dirname(parent_directory)
    root, extension = os.path.splitext(file_name)

    label_path = os.path.join(parent_directory, label_src_dir_name, f'{root}.json')

    return os.path.exists(label_path)
