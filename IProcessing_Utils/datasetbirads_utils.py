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
        
        label_1 = self.dataset['breast_birads_CC'].iloc[idx]
        label_2 = self.dataset['breast_birads_MLO'].iloc[idx]
        
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
            label_1 = self.dataset['breast_birads_CC'].iloc[idx - 1]
            label_2 = self.dataset['breast_birads_MLO'].iloc[idx - 1]
            
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
        label_1 = int(self.dataset['breast_birads_CC'].iloc[idx])
        label_2 = int(self.dataset['breast_birads_MLO'].iloc[idx])
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
    def __init__(self, class_data_dir, birads_mapping, ratio=1.0, transforms=None):
        self.transform = transforms
        self.birads_mapping = birads_mapping
        if not (0.0 < ratio <= 1.0):
            raise ValueError("Ratio must be between 0.0 and 1.0")

        # Retrieve all synthetic images and corresponding ROIs across the 4 BIRADS levels
        birad_dirs = {
            "BIRADS 1": os.path.join(class_data_dir, "BIRADS 1"),
            "BIRADS 2": os.path.join(class_data_dir, "BIRADS 2"),
            "BIRADS 3": os.path.join(class_data_dir, "BIRADS 3"),
            "BIRADS 4": os.path.join(class_data_dir, "BIRADS 4")
        }
        
        # Load images and organize them by label
        self.image_pairs = []
        for birad_name, birad_dir in birad_dirs.items():
            birad_images = sorted([os.path.join(birad_dir, img) for img in os.listdir(birad_dir) if img.endswith(".jpg")])
            
            # Calculate the number of images to process based on the ratio
            num_images_to_process = int(len(birad_images) * ratio)
            sampled_images = random.sample(birad_images, num_images_to_process)
            
            # Create pairs of images with the same label
            if len(sampled_images) % 2 != 0:
                sampled_images.pop()  # Ensure even number of images to form pairs
            
            pairs = [(sampled_images[i], sampled_images[i + 1], birad_name) for i in range(0, len(sampled_images), 2)]
            self.image_pairs.extend(pairs)

        # Shuffle the entire dataset
        random.shuffle(self.image_pairs)

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        dict_data = {}
        is_synth = [True, True]

        # Load a pair of images and their shared label
        img_path_1, img_path_2, birad_name = self.image_pairs[idx]

        # Map the BIRADS level to a numerical label
        label = list(self.birads_mapping.keys()).index(birad_name.replace(' ', ''))

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
        dict_data['labels'] = {'label1': torch.tensor(label, dtype=torch.float32),
                               'label2': torch.tensor(label, dtype=torch.float32)}
        dict_data['is_synth'] = is_synth

        return dict_data
