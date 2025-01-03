import torch
import cv2
import os
import logging
from PIL import Image
import numpy as np


def torch_CountUpContinuingOnes(b_arr):
    left = torch.arange(len(b_arr))
    left[b_arr > 0] = 0
    left = torch.cummax(left, dim=-1)[0]

    rev_arr = torch.flip(b_arr, [-1])
    right = torch.arange(len(rev_arr))
    right[rev_arr > 0] = 0
    right = torch.cummax(right, dim=-1)[0]
    right = len(rev_arr) - 1 - torch.flip(right, [-1])

    return right - left - 1

def torch_ExtractBreast_with_padding_single_side(img_ori, target_size=(512, 512), padding=1):
    # Detect background and set to zero
    img = torch.where(img_ori <= 20, torch.zeros_like(img_ori), img_ori)
    height, _ = img.shape

    # Extract the main breast region (same as before)
    y_a = height // 2 + int(height * 0.4)
    y_b = height // 2 - int(height * 0.4)
    b_arr = img[y_b:y_a].to(torch.float32).std(dim=0) != 0
    continuing_ones = torch_CountUpContinuingOnes(b_arr)
    col_ind = torch.where(continuing_ones == continuing_ones.max())[0]
    img = img[:, col_ind]

    _, width = img.shape
    x_a = width // 2 + int(width * 0.4)
    x_b = width // 2 - int(width * 0.4)
    b_arr = img[:, x_b:x_a].to(torch.float32).std(dim=1) != 0
    continuing_ones = torch_CountUpContinuingOnes(b_arr)
    row_ind = torch.where(continuing_ones == continuing_ones.max())[0]
    breast_region = img_ori[row_ind][:, col_ind]

    # Resize the extracted breast region while maintaining the aspect ratio
    breast_height, breast_width = breast_region.shape
    aspect_ratio = breast_width / breast_height

    # Define target dimensions based on aspect ratio
    if aspect_ratio > 1:
        # Wider than tall
        new_width = target_size[1] - padding
        new_height = int(new_width / aspect_ratio)
    else:
        # Taller than wide
        new_height = target_size[0] - padding
        new_width = int(new_height * aspect_ratio)

    resized_breast = cv2.resize(breast_region.cpu().numpy(), (new_width, new_height))
    resized_breast = torch.from_numpy(resized_breast)

    # Determine which side has lower intensity
    pad_x = target_size[1] - new_width
    pad_y = target_size[0] - new_height

    # Initialize offsets
    x_offset = 0
    y_offset = 0

    # Decide padding side for x-axis
    if pad_x > 0:
        left_intensity = resized_breast[:, 0].mean()
        right_intensity = resized_breast[:, -1].mean()
        if left_intensity < right_intensity:
            # Pad on the left side
            x_offset = pad_x
        else:
            # Pad on the right side
            x_offset = 0

    # Decide padding side for y-axis
    if pad_y > 0:
        top_intensity = resized_breast[0, :].mean()
        bottom_intensity = resized_breast[-1, :].mean()
        if top_intensity < bottom_intensity:
            # Pad on the top side
            y_offset = pad_y
        else:
            # Pad on the bottom side
            y_offset = 0

    # Create a padded image with the target size and place the resized breast region
    padded_img = torch.zeros(target_size, dtype=resized_breast.dtype)
    padded_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_breast

    return padded_img

def resize_with_padding_vini(img_np, target_size=(512, 512), padding=1):
    # Convert image to a torch tensor if not already
    img_torch = torch.from_numpy(img_np).to(torch.float32)

    # Ensure the breast region fits within a consistent scale
    breast_height, breast_width = img_torch.shape
    aspect_ratio = breast_width / breast_height

    # Define target dimensions based on aspect ratio
    if aspect_ratio > 1:
        # Wider than tall
        new_width = target_size[1] - padding
        new_height = int(new_width / aspect_ratio)
    else:
        # Taller than wide
        new_height = target_size[0] - padding
        new_width = int(new_height * aspect_ratio)
    
    # Resize while maintaining the aspect ratio
    resized_breast = cv2.resize(img_np, (new_width, new_height))
    resized_breast = torch.from_numpy(resized_breast)

     # Determine which side has lower intensity
    pad_x = target_size[1] - new_width
    pad_y = target_size[0] - new_height

    # Initialize offsets
    x_offset = 0
    y_offset = 0

    # Decide padding side for x-axis
    if pad_x > 0:
        left_intensity = resized_breast[:, 0].mean()
        right_intensity = resized_breast[:, -1].mean()
        if left_intensity < right_intensity:
            # Pad on the left side
            x_offset = pad_x
        else:
            # Pad on the right side
            x_offset = 0

    # Decide padding side for y-axis
    if pad_y > 0:
        top_intensity = resized_breast[0, :].mean()
        bottom_intensity = resized_breast[-1, :].mean()
        if top_intensity < bottom_intensity:
            # Pad on the top side
            y_offset = pad_y
        else:
            # Pad on the bottom side
            y_offset = 0

    # Create a padded image with the target size and place the resized breast region
    padded_img = torch.zeros(target_size, dtype=resized_breast.dtype)
    padded_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_breast

    return padded_img

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_img = clahe.apply(image)
    return equalized_img

def mean_variance_normalization(img_torch):
    img_min, img_max = img_torch.min(), img_torch.max()
    img_torch = (img_torch - img_min) / (img_max - img_min) * 255
    return img_torch

def resize_image(image_path, size=(512, 512)):
    if os.path.exists(image_path):
        img_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Check if the image was read successfully
        if img_np is None:
            logging.error(f"Failed to load image at {image_path}.")
            return None  # You could return a default image or raise an error
            
        if 'emory' in image_path:
            img_torch = torch.from_numpy(img_np).to(torch.float32)
            img_torch = mean_variance_normalization(img_torch)

            # Extract the breast region
            breast_region = torch_ExtractBreast_with_padding_single_side(img_torch)

            # Convert the result back to a NumPy array for visualization or further processing
            breast_region_np = breast_region.cpu().numpy().astype(np.uint8)
            # breast_region_np = apply_clahe(breast_region_np)
            breast_region_rgb = np.repeat(breast_region_np[:, :, np.newaxis], 3, axis=2)

            # Convert the NumPy array to a PIL image for compatibility with transforms
            breast_region_pil = Image.fromarray(breast_region_rgb)
        else:
            # img_torch = torch.from_numpy(img_np).to(torch.float32)
            # # Normalize the image
            img_np = mean_variance_normalization(img_np)
            breast_region_np = resize_with_padding_vini(img_np)
            # Convert the result back to a NumPy array for visualization or further processing
            breast_region_np = breast_region_np.cpu().numpy().astype(np.uint8)
            # breast_region_np = apply_clahe(breast_region_np)

            breast_region_rgb = np.repeat(breast_region_np[:, :, np.newaxis], 3, axis=2)

            # Convert the NumPy array to a PIL image for compatibility with transforms
            breast_region_pil = Image.fromarray(breast_region_rgb)

        img = breast_region_pil
        return img
    else:
        logging.error(f"Image path does not exist: {image_path}")
        return None
