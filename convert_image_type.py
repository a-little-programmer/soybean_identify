import os
import cv2

from common import *

def convert_images_to_jpg(directory, delete_old_file=False):
    # Iterate over all files and subdirectories in the directory
    for root, _, files in os.walk(directory):
        print(root)
        for filename in files:
            if filename.lower().endswith('.bmp'):
                # Construct full file path
                old_path = os.path.join(root, filename)
                
                # Read the image file using OpenCV
                image = cv2.imread(old_path)
                
                # Define the path for the .jpg file with the same base name
                new_path = os.path.join(root, f'{os.path.splitext(filename)[0]}.png')
                
                # Write the image as a .jpg file
                cv2.imwrite(new_path, image)
                
                if delete_old_file:
                    # Remove the original file
                    os.remove(old_path)

if __name__ == '__main__':
    # Specify the directory containing the .bmp files
    directory = '/Users/hjx/Documents/research/data-label/blood-tube/images-extra'

    convert_images_to_jpg(directory, True)

    print("Conversion completed.")
