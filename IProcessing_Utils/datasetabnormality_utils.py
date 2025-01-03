import os
import random
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger(__name__)
from IProcessing_Utils.extract_breast import *

class CustomDatasetWithROI(Dataset):
    def __init__(self, data_set: pd.DataFrame, yolo_model, transform=None):
        self.dataset = data_set
        self.transform = transform
        self.yolo_model = yolo_model  # Pass the trained YOLO model during initialization

    def __len__(self):
        return len(self.dataset['patient_id'])

    def __getitem__(self, idx):
        dict_data = {}
        
        # Retrieve paths
        image_path_1 = self.dataset['image_path_CC'].iloc[idx]
        image_path_2 = self.dataset['image_path_MLO'].iloc[idx]
        
        label_1 = self.dataset['category_CC'].iloc[idx]
        label_2 = self.dataset['category_MLO'].iloc[idx]
        
        try:
            # Load images as grayscale, then convert to RGB
            image_1 = Image.open(image_path_1).convert('L').convert('RGB')
            image_2 = Image.open(image_path_2).convert('L').convert('RGB')
            
            # Ensure images are numpy arrays
            image_1 = np.array(image_1)
            image_2 = np.array(image_2)
        except OSError:
            print(f"Error loading image at index {idx}. Replacing with previous valid image.")
            # Load the previous valid image as a fallback
            image_path_1 = self.dataset['image_path_CC'].iloc[idx - 1]
            image_path_2 = self.dataset['image_path_MLO'].iloc[idx - 1]
            label_1 = self.dataset['category_CC'].iloc[idx - 1]
            label_2 = self.dataset['category_MLO'].iloc[idx - 1]
            
            image_1 = Image.open(image_path_1).convert('L').convert('RGB')
            image_2 = Image.open(image_path_2).convert('L').convert('RGB')
            image_1 = np.array(image_1)
            image_2 = np.array(image_2)
        
        # Use YOLO to extract ROIs
        roi_1 = self._extract_roi(image_1)
        roi_2 = self._extract_roi(image_2)
        
        # Apply transformations if provided
        if self.transform is not None:
            roi_1 = self.transform(image=roi_1)['image']
            roi_2 = self.transform(image=roi_2)['image']
            dict_data.update({'images': {'roi1': roi_1, 'roi2': roi_2}})
        else:
            dict_data.update({'images': {'roi1': roi_1, 'roi2': roi_2}})
        
        # Convert labels to tensors
        label1_tensor = torch.tensor(float(label_1), dtype=torch.float32)
        label2_tensor = torch.tensor(float(label_2), dtype=torch.float32)
        
        dict_data.update({'labels': {'label1': label1_tensor, 'label2': label2_tensor}})
        
        return dict_data
    
    def _extract_roi(self, image):
        """Run YOLO inference and extract ROI."""
        # Perform inference
        results = self.yolo_model(image, conf=0.5)
        
        # Get the bounding box with the highest confidence
        boxes = results[0].boxes  # Extract boxes from the results
        if len(boxes) > 0:
            box = boxes[0]  # Select the first (highest confidence) box
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            roi = image[y1:y2, x1:x2]  # Extract ROI
        else:
            # If no bounding box detected, return the full image as fallback
            roi = image
        
        return roi


class CustomDataset(Dataset):
    def __init__(self, data_set: pd.DataFrame, transforms=None):
        self.dataset = data_set
        self.transform = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dict_data = {}
        is_synth = [False, False]
        
        # Retrieve paths
        label_1 = int(self.dataset['category_CC'].iloc[idx])
        label_2 = int(self.dataset['category_MLO'].iloc[idx])
        image_path_1 = self.dataset['image_path_CC'].iloc[idx]
        image_path_2 = self.dataset['image_path_MLO'].iloc[idx]
        
        img_1 = resize_image(image_path_1)
        img_2 = resize_image(image_path_2)
        
        if img_1 is None:
            logger.warning(f"Image loading failed for index {idx}. Using fallback image.")
            img_1 = resize_image(self.dataset['image_path_CC'].iloc[max(0, idx - 1)])
        if img_2 is None:
            logger.warning(f"Image loading failed for index {idx}. Using fallback image.")
            img_2 = resize_image(self.dataset['image_path_MLO'].iloc[max(0, idx - 1)])
            
        # Apply transformations if images are loaded
        if img_1:
            img_1 = np.array(img_1)
            img_1 = self.transform(image=img_1)['image'] if self.transform else img_1
        if img_2:
            img_2 = np.array(img_2)
            img_2 = self.transform(image=img_2)['image'] if self.transform else img_2

        # Store images, labels, and synthetic tracker
        dict_data['images'] = {'img1': img_1, 'img2': img_2}
        dict_data['labels'] = {'label1': torch.tensor(label_1, dtype=torch.float32),
                               'label2': torch.tensor(label_2, dtype=torch.float32)}
        dict_data['is_synth'] = is_synth
        
        return dict_data

class CustomSynthDataset(Dataset):
    def __init__(self, class_data_dir, cat_map, ratio=1.0, transforms=None):
        self.cat_map = cat_map
        self.transform = transforms
        if not (0.0 < ratio <= 1.0):
            raise ValueError("Ratio must be between 0.0 and 1.0")

        # Retrieve all synthetic images and corresponding ROIs across the 3 lesions
        lesion_dirs = {
            "Asymmetry": os.path.join(class_data_dir, "asymmetry"),
            "Suspicious Calcification": os.path.join(class_data_dir, "suspicious calcification"),
            "Mass": os.path.join(class_data_dir, "mass")
        }
        
        # Load images and shuffle
        self.image_pairs = []
        for lesion_name, lesion_dir in lesion_dirs.items():
            lesion_images = sorted([os.path.join(lesion_dir, img) for img in os.listdir(lesion_dir) if img.endswith(".png")])
            
            # Calculate the number of images to process based on the ratio
            num_images_to_process = int(len(lesion_images) * ratio)
            sampled_images = random.sample(lesion_images, num_images_to_process)
            self.image_pairs.extend([(img, lesion_name) for img in sampled_images])

        # Shuffle the entire dataset
        random.shuffle(self.image_pairs)

        # Split the dataset into two halves
        midpoint = len(self.image_pairs) // 2
        self.first_half = self.image_pairs[:midpoint]
        self.second_half = self.image_pairs[midpoint:]

        if len(self.first_half) != len(self.second_half):
            print("Warning: Dataset size is odd, dropping the extra image.")

    def __len__(self):
        # The length is determined by the smaller of the two halves
        return min(len(self.first_half), len(self.second_half))

    def __getitem__(self, idx):
        dict_data = {}
        is_synth = [True, True]

        # Load one image from each half
        img_path_1, lesion_name_1 = self.first_half[idx]
        img_path_2, lesion_name_2 = self.second_half[idx]

        label_1 = list(self.cat_map.keys()).index(lesion_name_1)
        label_2 = list(self.cat_map.keys()).index(lesion_name_2)

        img_1 = Image.open(img_path_1)
        img_2 = Image.open(img_path_2)

        # Apply transformations if specified
        if img_1:
            img_1 = np.array(img_1)
            img_1 = self.transform(image=img_1)['image'] if self.transform else img_1
        if img_2:
            img_2 = np.array(img_2)
            img_2 = self.transform(image=img_2)['image'] if self.transform else img_2

        # Store images, labels, and synthetic tracker
        dict_data['images'] = {'img1': img_1, 'img2': img_2}
        dict_data['labels'] = {'label1': torch.tensor(label_1, dtype=torch.float32),
                               'label2': torch.tensor(label_2, dtype=torch.float32)}
        dict_data['is_synth'] = is_synth

        return dict_data