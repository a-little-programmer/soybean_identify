import cv2
import numpy as np
import argparse

def calculate_optical_flow(img1, img2):
    """Calculate optical flow and return the average motion magnitude."""
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    motion_magnitude = np.mean(magnitude)
    return motion_magnitude

def is_similar_image(image_path1, image_path2, motion_threshold=1.0):
    """Check if there is motion between two images based on optical flow and return True or False."""
    # Read the images
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Error: One of the images could not be loaded.")
        return None

    # Apply histogram equalization
    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)

    # Resize to the same dimensions (if needed)
    img1 = cv2.resize(img1, (640, 480))
    img2 = cv2.resize(img2, (640, 480))

    # Compute optical flow
    motion_magnitude = calculate_optical_flow(img1, img2)

    return motion_magnitude < motion_threshold

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Check if the assembly line is moving or still based on optical flow.")
    parser.add_argument("image1", help="Path to the first image.")
    parser.add_argument("image2", help="Path to the second image.")
    parser.add_argument("--motion_threshold", type=float, default=1.0, help="Threshold for motion magnitude (default: 1.0).")

    # Parse arguments
    args = parser.parse_args()

    # Check if there is motion between the images
    result = is_similar_image(args.image1, args.image2, motion_threshold=args.motion_threshold)

    if result is not None:
        print(f"The images are {'similar' if result else 'different'}.")

if __name__ == "__main__":
    main()
