import os
import copy
from typing import Union, List, Literal, Dict, Callable, Tuple

import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns
from tqdm.auto import tqdm

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger(__name__)

class GeneralizedDiceLoss(torch.nn.Module):
    def __init__(self, use_multiclass=True):
        super(GeneralizedDiceLoss, self).__init__()
        self.multiclass = use_multiclass

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a softmax or equivalent activation layer
        if self.multiclass:
            inputs = F.softmax(inputs, dim=1)
        else:
            inputs = F.sigmoid(inputs) 
            
        targets = torch.nn.functional.one_hot(targets, num_classes=inputs.shape[1])
        # targets = targets.permute(0, 3, 1, 2)  # Rearrange to match inputs (batch, classes, height, width)

        # flatten label and prediction tensors
        inputs = inputs.view(inputs.shape[0], inputs.shape[1], -1)
        targets = targets.view(targets.shape[0], targets.shape[1], -1)

        # intersection is equivalent to True Positive count
        intersection = torch.sum(inputs * targets, dim=2)
        
        # compute the Dice coefficient for each class
        dice_scores = (2. * intersection + smooth) / (torch.sum(inputs, dim=2) + torch.sum(targets, dim=2) + smooth)

        # average the Dice scores over all classes
        dice_loss = 1 - torch.mean(dice_scores, dim=1)

        return dice_loss.mean()

def get_loss_coefficients(synthetic_proportion):
    """
    Calculate loss coefficients based on proportion of synthetic data.
    
    Args:
        synthetic_proportion (float): Proportion of synthetic data (0.0 to 1.0)
    
    Returns:
        tuple: (original_coefficient, synthetic_coefficient)
    """
    k = 0.8  # scaling factor
    original_coef = 1.0
    synthetic_coef = k * (synthetic_proportion / (1 + synthetic_proportion))
    
    return original_coef, synthetic_coef

def run_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: Callable, 
    optimizer: optim.Optimizer = None,
    scaler: torch.cuda.amp.GradScaler = None, 
    phase: str = 'train', 
    synth_prop=0.25, 
    num_classes=3, 
    device='cuda') -> dict:
    """
    Runs a single epoch of training or validation with mixed precision.

    Args:
        model: The model being trained or validated.
        dataloader: The dataloader for training or validation.
        criterion: Loss function.
        optimizer: Optimizer (if phase is 'train').
        scaler: GradScaler for mixed precision training.
        phase: Specifies 'train' or 'val' phase.
        device: The device to run the model on.

    Returns:
        A dictionary containing loss and accuracy metrics.
    """
    
    running_loss, running_corrects, ns = 0.0, 0, 0
    precision_metric = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
    recall_metric = MulticlassRecall(num_classes=num_classes, average='macro').to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    
    for batch in tqdm(dataloader, desc=f"{phase.capitalize()} Epoch"):
        images, labels = batch['images'], batch['labels']
        images = {k: v.to(device) for k, v in images.items()}
        label1, label2 = labels['label1'].to(device), labels['label2'].to(device)
        y = torch.max(label1, label2).long()
        
        if phase=="train":
            combined_is_synth = batch['is_synth'][0] | batch['is_synth'][1]
            true_indices_combined = torch.where(combined_is_synth)[0]
            # synth_labels = y[true_indices_combined]
            orig_labels = y[~true_indices_combined]
        
        
        with torch.cuda.amp.autocast():  # Enable mixed precision
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            # loss = criterion(outputs, y)
            if phase=="train":
                # synth_loss = criterion(outputs[true_indices_combined], synth_labels)
                orig_loss = criterion(outputs[~true_indices_combined], orig_labels)
                
                # orig_coef, synth_coef = get_loss_coefficients(synth_prop)
                # loss = orig_coef * orig_loss + synth_coef * synth_loss
                loss = orig_loss
            else:
                loss = criterion(outputs, y)
        
        if phase == 'train' and optimizer:
            optimizer.zero_grad()
            scaler.scale(loss).backward()  # Scale the loss to handle mixed precision gradients
            scaler.step(optimizer)         # Update weights
            scaler.update()                # Update the scaler for the next iteration
        
        # Update metrics
        precision_metric.update(preds, y)
        recall_metric.update(preds, y)
        f1_metric.update(preds, y)

        # Accumulate metrics
        running_loss += loss.item() * images['img1'].size(0)
        running_corrects += torch.sum(preds == y.data)
        ns += preds.size(0)
    
    epoch_loss = running_loss / ns
    epoch_acc = running_corrects.double() / ns
    epoch_precision = precision_metric.compute().item()
    epoch_recall = recall_metric.compute().item()
    epoch_f1 = f1_metric.compute().item()
    
    # Reset metrics
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc.item(),
        'precision': epoch_precision,
        'recall': epoch_recall,
        'f1_score': epoch_f1
    }

def train_model(
    model: nn.Module, 
    criterion: Callable, 
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler, 
    loaders: dict, 
    synth_prop=0.25, 
    num_epochs: int = 10,
    patience: int = 5,
    num_classes: int = 3,
    device='cuda') -> Tuple[nn.Module, List[float], List[float]]:  
    """
    Train a model on a dataset with mixed precision.

    Args:
        model: The model to train.
        criterion: The loss function.
        optimizer: The optimizer.
        scheduler: The learning rate scheduler.
        loaders: Dictionary containing training and validation dataloaders.
        alpha: Weight for synthetic loss (if applicable).
        num_epochs: The number of epochs to train for.
        patience: Early stopping patience.
        device: The device to run the model on.

    Returns:
        A tuple containing the trained model, list of training losses, and list of validation losses.
    """
    
    # Initialize best model weights, metrics, and early stopping variables
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    counter = 0
    train_loss, valid_loss = [], []
    scaler = torch.cuda.amp.GradScaler()  # Initialize GradScaler for mixed precision
    
    dirpath = os.path.join("abnormality_runs", f"3classes_{synth_prop}")
    best_modelpath = os.path.join(dirpath, "EfficientNet_mammo.pth.tar")
    metricspath = os.path.join(dirpath, "results.csv")
    os.makedirs(dirpath, exist_ok=True)
    metrics_df = pd.DataFrame()
        

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training Phase
        model.train()
        train_metrics = run_epoch(
            model, 
            loaders['train'], 
            criterion, 
            optimizer, 
            scaler, 
            phase='train', 
            synth_prop=synth_prop, 
            num_classes=num_classes,
            device=device)
        train_loss.append(train_metrics['loss'])
        
        # Validation Phase
        model.eval()
        with torch.no_grad():
            val_metrics = run_epoch(
                model, 
                loaders['val'], 
                criterion, 
                optimizer=None, 
                scaler=None, 
                phase='val', 
                num_classes=num_classes,
                device=device)
        valid_loss.append(val_metrics['loss'])
        
        train_df = pd.DataFrame([train_metrics]).add_prefix("train_")
        val_df = pd.DataFrame([val_metrics]).add_prefix("val_")

        epoch_metrics = pd.concat([train_df, val_df], axis=1)
        metrics_df = pd.concat([metrics_df, epoch_metrics], ignore_index=True)
        metrics_df.to_csv(metricspath, index=False)
        
        # Scheduler step based on validation loss
        # scheduler.step()
        scheduler.step(val_metrics['loss'])
        
        # Print epoch summary
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Early Stopping based on validation accuracy
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, best_modelpath)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping due to no improvement.")
                break
    
    print(f'Best validation accuracy: {best_acc:.4f}')
    metrics_df.to_csv(metricspath, index=False)
    model.load_state_dict(best_model_wts)
    return model, train_loss, valid_loss, metrics_df

def testpreds(model : nn.Module, test_dl : DataLoader, device) -> List:
    """
      Generates predictions for the test set.

      Args:
          model: The model to be evaluated.
          test_dl: The test dataloader.

      Returns:
          A list of predictions for the test set.
      """
    all_preds, all_labels = [], []
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        
        for dict_data in tqdm(test_dl, desc="testing"):
            images, labels = dict_data['images'], dict_data['labels']
            for img in images:
                images[img] = images[img].to(device)
            label1, label2 = labels['label1'].to(device), labels['label2'].to(device)
            y = torch.max(label1, label2)

            outp = model(images)

            _, pred = torch.max(outp, 1)
            num_samples += pred.shape[0]
            num_correct += torch.sum(pred == y.data)

            # add to lists
            all_preds.append(pred.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())
        
    print(
        f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
    )
    return np.concatenate(all_preds, axis=0), np.concatenate(all_labels, axis=0)

def predict_image(model:nn.Module, loader:DataLoader, class_names, device, count_images:int=3) -> None:
    model.eval()
    fig, axs = plt.subplots(1, count_images, figsize=(15, 10))
    
    num_images = count_images
    image_count = 0
    
    for dict_data in loader:
        images, labels = dict_data['images'], dict_data['labels']
        for img in images:
            images[img] = images[img].to(device)
        label1, label2 = labels['label1'].to(device), labels['label2'].to(device)
        y = torch.max(label1, label2)
        
        with torch.no_grad():
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
       
            if image_count < num_images:

                image = images['img1'][image_count].permute(1, 2, 0).cpu().numpy()
                true_label = int(y[image_count].detach().cpu().item())
                predicted_label = int(predictions[image_count].detach().cpu().item())

                axs[image_count].imshow(image)
                axs[image_count].set_title(f'True: ({class_names[true_label]}),\n Predicted: ({class_names[predicted_label]})')
                axs[image_count].axis('off')

                image_count += 1
            else:
                break
        
        if image_count >= num_images:
            break
    
    plt.show()

def plot_metrics(metrics_df):
    """
    Create a figure containing multiple plots based on the metrics DataFrame.
    
    Args:
        metrics_df: DataFrame containing training and validation metrics with prefixed column names.
    """
    # Extract data
    train_loss = metrics_df['train_loss']
    val_loss = metrics_df['val_loss']
    train_acc = metrics_df['train_accuracy']
    val_acc = metrics_df['val_accuracy']
    train_f1 = metrics_df['train_f1_score']
    val_f1 = metrics_df['val_f1_score']
    train_precision = metrics_df['train_precision']
    train_recall = metrics_df['train_recall']
    val_precision = metrics_df['val_precision']
    val_recall = metrics_df['val_recall']
    epochs = range(1, len(metrics_df) + 1)  # 1-based epoch index

    # Create the figure and subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle("Training and Validation Metrics", fontsize=16)

    # Train Loss vs Val Loss
    axs[0, 0].plot(epochs, train_loss, label="Train Loss", marker='o')
    axs[0, 0].plot(epochs, val_loss, label="Val Loss", marker='o')
    axs[0, 0].set_title("Train Loss vs Val Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()

    # Train Acc vs Val Acc
    axs[0, 1].plot(epochs, train_acc, label="Train Accuracy", marker='o')
    axs[0, 1].plot(epochs, val_acc, label="Val Accuracy", marker='o')
    axs[0, 1].set_title("Train Accuracy vs Val Accuracy")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Accuracy")
    axs[0, 1].legend()

    # Train F1 Score vs Val F1 Score
    axs[1, 0].plot(epochs, train_f1, label="Train F1-Score", marker='o')
    axs[1, 0].plot(epochs, val_f1, label="Val F1-Score", marker='o')
    axs[1, 0].set_title("Train F1-Score vs Val F1-Score")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("F1-Score")
    axs[1, 0].legend()

    # Train Precision vs Train Recall
    axs[1, 1].plot(epochs, train_precision, label="Train Precision", marker='o')
    axs[1, 1].plot(epochs, val_precision, label="Val Precision", marker='o')
    axs[1, 1].set_title("Train Precision vs Val Precision")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Precision")
    axs[1, 1].legend()

    # Val Precision vs Val Recall
    axs[2, 0].plot(epochs, train_precision, label="Train Precision", marker='o')
    axs[2, 0].plot(epochs, val_precision, label="Val Precision", marker='o')
    axs[2, 0].set_title("Train Recall vs Val Recall")
    axs[2, 0].set_xlabel("Epoch")
    axs[2, 0].set_ylabel("Recall")
    axs[2, 0].legend()

    # Adjust layout
    axs[2, 1].axis('off')  # Leave the bottom-right plot empty
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def calculate_metrics_save_csv(y_true, y_scores, classes, synth_prop=0.25):
    """
    Calculate metrics (accuracy, precision, recall, F1-score) for each class and save to CSV.
    
    Args:
        y_true: Ground truth labels.
        y_scores: Predicted scores or probabilities.
        classes: List of class names.
        csv_filename: Name of the CSV file to save metrics.
        
    Returns:
        DataFrame containing metrics for each class.
    """
    # Initialize dictionary for storing metrics
    metrics = {}
    preds = y_scores
    
    for i, class_name in enumerate(classes):
        class_indices = (y_true == i)  # True if the ground truth is the current class
        metrics[f"{class_name}_acc"] = accuracy_score(y_true[class_indices], preds[class_indices])
        metrics[f"{class_name}_prec"] = precision_score(y_true, preds, labels=[i], average='macro')
        metrics[f"{class_name}_rec"] = recall_score(y_true, preds, labels=[i], average='macro')
        metrics[f"{class_name}_f1score"] = f1_score(y_true, preds, labels=[i], average='macro')
    
    # Save to CSV
    metrics_df = pd.DataFrame([metrics])
    dirpath = os.path.join("abnormality_runs", f"3classes_{synth_prop}")
    metricspath = os.path.join(dirpath, "test_results.csv")
    metrics_df.to_csv(metricspath, index=False)
    print(f"Metrics saved to {metricspath}")
    return metrics_df

def plot_classification_report(y_true, y_pred, classes):
    """
    Plot a heatmap of classification metrics.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        classes: List of class names.
    """
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    df = pd.DataFrame(report).transpose()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.iloc[:-2, :-1].T, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('Classification Metrics Heatmap')
    plt.tight_layout()
    plt.show()
    
def plot_confusion_matrix(cm, classes):
    """
    Plot a confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix.
        classes: List of class names.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, 
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()