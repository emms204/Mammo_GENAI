
# Mammo_GENAI

## Overview

Welcome to the **Mammo_GENAI** repository. This project explores the integration of **Generative AI (GENAI)** and **computer vision** for advanced **mammogram analysis**. It aims to leverage synthetic data generation, object detection, abnormality classification, and lesion inpainting techniques to improve the accuracy and efficiency of breast cancer diagnosis. This repository contains a series of Jupyter notebooks and utilities designed for various tasks in mammogram image analysis, including:

- **Object Detection** using YOLO-based models
- **BIRADS Classification** for categorizing mammogram findings
- **Abnormality Classification** to identify and classify anomalies in mammograms
- **Lesion Inpainting** for generating realistic lesions in synthetic data
- **Data Preprocessing & Utility Scripts** for converting medical images (e.g., DICOM to PNG)
- **Inference Notebooks** for running models and generating predictions

This project is built with the objective to explore **Hybrid Generative AI** techniques for improving the diagnostic process in breast cancer detection.

## Contents

The repository includes the following directories and files:

### 1. **Object Detection**
- **`vini_yolo_real.ipynb`**: YOLO-based object detection model for real mammogram data.
- **`vini_yolo_synth.ipynb`**: YOLO-based object detection model using synthetic data.
- **`combined_yolo_detection_without_synthetic_data.ipynb`**: Combined detection model excluding synthetic data.
- **`emory_yolo_real.ipynb`**: YOLO-based model for real data from the Emory dataset.
- **`emory_yolo_synth.ipynb`**: YOLO-based model using synthetic data for the Emory dataset.
- **`vindr_object_detection_using_2classes.ipynb`**: Object detection for two classes in the VinDr dataset.

### 2. **Dreambooth**
- **`dreambooth_mammo_lesion_inpainting.ipynb`**: Generating synthetic lesion images using Dreambooth for lesion inpainting.
- **`dreambooth_mammo_lora_ft.ipynb`**: Fine-tuning Dreambooth for mammogram lesion generation using LoRA (Low-Rank Adaptation).
- **`dreambooth_mammo_pt.ipynb`**: Dreambooth model fine-tuning for mammogram lesion generation.
- **`dreambooth_mammo.ipynb`**: General-purpose Dreambooth model for mammogram analysis.

### 3. **BIRADS Classification**
- **`birads_classification_with_synthetic_data.ipynb`**: BIRADS classification model trained on synthetic data.
- **`birads_model_classification.ipynb`**: BIRADS classification using a real-world model.
- **`birads_classification_without_synthetic_data.ipynb`**: BIRADS classification excluding synthetic data.

### 4. **Abnormality Classification**
- **`abnormality_classification_with_synthetic_data.ipynb`**: Abnormality classification using synthetic data.
- **`abnormality_model_classification.ipynb`**: Abnormality classification using a real-world model.
- **`abnormality_classification_without_synthetic_data.ipynb`**: Abnormality classification excluding synthetic data.

### 5. **Image Processing Utilities**
- **`DCM_to_PNG.ipynb`**: Utility for converting DICOM medical images to PNG format.

### 6. **Data Analysis**
- **`emory_mammo_eda.ipynb`**: Exploratory data analysis (EDA) on the Emory mammogram dataset.
- **`VinDR_Mammogram_EDA.ipynb`**: Exploratory data analysis (EDA) on the VinDr dataset.

### 7. **Hybrid Generative AI Discussion**
- **`Hybrid_Generative_AI_Final_Draft.pdf`**: Paper discussing the integration of **Generative AI** with medical imaging for breast cancer diagnosis, highlighting the potential of synthetic data generation and hybrid models in improving detection accuracy.

### 8. **Inference**
- **`Inference.ipynb`**: A notebook for performing inference using trained models for object detection, classification, and lesion inpainting.

### Requirements

- Python 3.10

## Results & Discussion

The results from various notebooks will allow you to evaluate different models and techniques for mammogram image analysis. Key findings include:

- **Object Detection**: Models trained on real and synthetic data provide insights into the accuracy and generalization of YOLO-based object detection.
- **BIRADS Classification**: The impact of synthetic data on classification performance is evaluated in terms of model robustness and accuracy.
- **Abnormality Classification**: Analyzing the role of synthetic data in identifying abnormalities and improving model performance for breast cancer detection.
- **Lesion Inpainting**: Demonstrating the ability of Generative AI models to generate realistic lesions in synthetic datasets, which can be used to enhance model training and improve diagnosis.

## Contributions

We welcome contributions to improve the models and expand the project. If you wish to contribute, please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

