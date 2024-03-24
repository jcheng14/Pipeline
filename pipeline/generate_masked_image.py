import argparse
import os
import glob
import cv2

import os
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Load segment masks
def load_sam_masks(path):
    # Initialize a list to store mask data
    masks_data = []
    
    # Read metadata from CSV
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, mode='r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            # Convert string values to appropriate types
            mask_info = {
                'id': int(row['id']),
                'area': float(row['area']),
                'bbox': [float(row['bbox_x0']), float(row['bbox_y0']), float(row['bbox_w']), float(row['bbox_h'])],
                'point_input': [float(row['point_input_x']), float(row['point_input_y'])],
                'predicted_iou': float(row['predicted_iou']),
                'stability_score': float(row['stability_score']),
                'crop_box': [float(row['crop_box_x0']), float(row['crop_box_y0']), float(row['crop_box_w']), float(row['crop_box_h'])]
            }
            masks_data.append(mask_info)
    
    # Read each mask image
    for mask_data in masks_data:
        filename = f"{mask_data['id']}.png"
        mask_path = os.path.join(path, filename)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        if mask_image is not None:
            mask_data['segmentation'] = mask_image / 255.0  # Normalize to be 0 and 1
    
    return masks_data


def is_wood_by_rule(x, y, w, h):
    #todo:
    if x <800 or x > 1700:
        return False
    if y > 1000:
        return False
    if w > 400:
        return False
    if h > 400:
        return False
    return True


def is_wood_by_model(detection, x, y, w, h):
    img_copy = detection.copy()
    img_copy = img_copy.astype(np.uint8)  #convert to an unsigned byte
    img_copy*=255
    '''
    # Apply Hough transform to greyscale image
    rows = img_copy.shape[0]
    circles = cv2.HoughCircles(img_copy, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                param1=100, param2=30,
                                minRadius=10, maxRadius=100)
    if circles is not None:
    draw_circle(img_copy, circles)
    return True
    return False
    '''
    contours, _ = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    distance = cv2.distanceTransform(img_copy, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, max_val, _, centre = cv2.minMaxLoc(distance)
    circle = cv2.circle(img_copy, centre, int(max_val), 0, 2)
    # cv2_imshow(circle)

    ellipses = []
    for cnt in contours:
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            # print(ellipse[1][0], ellipse[1][1], max_val)
            if 0.5 < ellipse[1][0]/ellipse[1][1] < 3 and 1 < max(ellipse[1][0], ellipse[1][1]) / max_val < 3.5:
                ellipses.append(ellipse)

    if ellipses is not None and len(ellipses) == 1:
    # if(max(ellipse[1][0], ellipse[1][1]) / max_val < 2):
    # uncomment if you want to see the logs
    # if y >= 988:
    # draw_ellipse(img_copy, ellipses)
        return True
    return False


def filter_circle_like_objects(mask, circularity_threshold=0.75):
    mask_copy = mask.copy()
    mask_copy = mask_copy.astype(np.uint8)  #convert to an unsigned byte
    mask_copy*=255

    # Find contours in the mask
    contours, _ = cv2.findContours(mask_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        # Avoid division by zero and ignore very small contours
        if perimeter == 0 or area < 1:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)

        if circularity >= circularity_threshold:
            # This mask contains a circle-like object
            # circle_like_masks.append(mask)
            return True

    return False


def resolve_overlapping_masks(masks):
    # Convert masks to binary (if they aren't already)
    binary_masks = [mask.astype(np.uint8) * 255 for mask in masks]

    # Record the area and index of each mask
    areas = [cv2.contourArea(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]) for mask in binary_masks]
    indices = list(range(len(binary_masks)))

    # Sort masks by area (smallest to largest)
    sorted_indices = sorted(indices, key=lambda i: areas[i])

    # Initialize an empty image to record which pixels are covered
    height, width = binary_masks[0].shape
    covered = np.zeros((height, width), dtype=np.uint8)

    # List to store the indices of the masks that we want to keep
    keep_masks = []

    for i in sorted_indices:
        # Check if current mask overlaps with any previously accepted mask
        overlap = cv2.bitwise_and(covered, binary_masks[i])
        if not np.any(overlap):  # If no overlap, this mask is kept
            keep_masks.append(i)
            # Update the 'covered' image
            covered = cv2.bitwise_or(covered, binary_masks[i])

    # Create a new list of masks to keep
    new_masks = [masks[i] for i in keep_masks]
    return new_masks


def draw_circle_like_masks_on_image(original_image, circle_like_masks):
    # Make a copy of the original image to draw on
    image_with_circles = original_image.copy()

    for mask in circle_like_masks:
        mask_copy = mask.copy()
        mask_copy = mask_copy.astype(np.uint8)  #convert to an unsigned byte
        mask_copy*=255

        # Find contours in the mask
        contours, _ = cv2.findContours(mask_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw each contour on the original image
        # You can choose a color (B, G, R) and thickness for the outline
        cv2.drawContours(image_with_circles, contours, -1, (0, 255, 0), 2)  # Green color, 2px thickness

    return image_with_circles


plotImage = False
def filter_masks(masks, image):
    #fitler masks
    circle_like_masks = []
    for detection in masks:
        # print(detection)
        x = detection['bbox'][0]
        y = detection['bbox'][1]
        w = detection['bbox'][2]
        h = detection['bbox'][3]
        # if not is_wood_by_rule(x, y, w, h):
        #     continue
        if is_wood_by_model(detection['segmentation'], x, y, w, h):
    
            if filter_circle_like_objects(detection['segmentation'], circularity_threshold=0.7):
                circle_like_masks.append(detection['segmentation'])
    
    new_masks = resolve_overlapping_masks(circle_like_masks)
    # new_image_with_circles = draw_circle_like_masks_on_image(masks, new_masks)
    
    # Initialize 'combined_mask' with the correct data type and size
    if new_masks:  # Check if the list is not empty
        combined_mask = np.zeros_like(new_masks[0], dtype=np.uint8)  # Ensure it's of type np.uint8
    
    for mask in new_masks:
        # Ensure the mask is not None and has content
        if mask is not None and mask.size > 0:
            # Ensure mask is in the correct format
            mask_copy = mask.astype(np.uint8)  # Convert to unsigned byte
            mask_copy = cv2.threshold(mask_copy, 0, 255, cv2.THRESH_BINARY)[1]  # Ensure binary format
    
            # Combine the mask with the cumulative one
            combined_mask = cv2.bitwise_or(combined_mask, mask_copy)
    
    if len(image.shape) == 3 and image.shape[-1] == 3:  # Check if the image is colored
        # Convert single-channel mask to three channels to match the original image
        combined_mask_colored = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
    else:
        combined_mask_colored = combined_mask  # No conversion needed for grayscale
    
    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, combined_mask_colored)

    return new_masks, masked_image


# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate masked images from segmentation results.')
parser.add_argument('--data', required=True, help='Directory containing the original images')
parser.add_argument('--seg_result', required=True, help='Directory containing the segmentation results')
parser.add_argument('--output', required=True, help='Directory where the masked images will be saved')
args = parser.parse_args()

data_path = args.data
seg_results_path = args.seg_result
output_path = args.output

if not os.path.exists(output_path):
    os.makedirs(output_path)

# List all PNG images in the data folder
image_files = glob.glob(os.path.join(data_path, '*.png'))

for image_file in image_files:
    # Read the left image
    top_image_left = cv2.imread(image_file)
    
    # Construct the corresponding segmentation result path
    base_name = os.path.basename(image_file)  # Extracts file name from path
    name_without_ext = os.path.splitext(base_name)[0]  # Remove the file extension
    seg_result_file = os.path.join(seg_results_path, name_without_ext)
    
    # Load the segmentation masks
    # Note: You need to define or import the function load_sam_masks
    top_masks_left = load_sam_masks(seg_result_file)
    
    # Filter the masks and apply them to the image
    # Note: You need to define or import the function filter_masks
    filtered_top_mask_left, masked_image_left = filter_masks(top_masks_left, top_image_left)
    
    # Construct the output file path and save the masked image
    output_file = os.path.join(output_path, f"masked_{base_name}")
    cv2.imwrite(output_file, masked_image_left)

    print(f"Processed and saved: {output_file}")

print("Done")
