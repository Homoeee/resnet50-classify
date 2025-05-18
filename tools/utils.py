from sklearn.metrics import average_precision_score
import numpy as np
import torch
import os
from torchvision import transforms
from dataset import VOC_CLASSES
from PIL import Image
import torch.nn as nn
import seaborn as sns
from sklearn.metrics import confusion_matrix
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt

def calculate_map(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            all_labels.append(labels.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())
    
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    
    # 计算每个类别的AP，然后取平均
    aps = []
    for i in range(len(VOC_CLASSES)):
        ap = average_precision_score(all_labels[:, i], all_preds[:, i])
        # print(f"AP for {VOC_CLASSES[i]}: {ap:.4f}")
        aps.append(ap)
    
    return np.mean(aps)

def predict_image(image_path, model, device, transform):
    """Predict classes for a single image"""
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probs = output.cpu().numpy()[0]
    
    # Get predicted classes (prob > 0.5) and their probabilities
    pred_classes = []
    pred_probs = []
    for i, prob in enumerate(probs):
        if prob > 0.5:
            pred_classes.append(VOC_CLASSES[i])
            pred_probs.append(prob)
    
    # Create visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    
    # Add prediction text
    prediction_text = "\n".join([f"{cls}: {prob:.2f}" for cls, prob in zip(pred_classes, pred_probs)])
    plt.text(10, 20, prediction_text, 
            color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save visualization
    output_path = os.path.join(os.path.dirname(image_path), 'prediction_result.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    return pred_classes, pred_probs

def plot_attention(image, att_map, pred_classes, true_classes=None):
    """Visualize attention heatmap over image"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Heatmap overlay
    ax2.imshow(image)
    ax2.imshow(att_map, cmap='jet', alpha=0.5)
    ax2.set_title('Attention Heatmap')
    ax2.axis('off')
    
    # Add prediction text
    pred_text = "Predicted: " + ", ".join(pred_classes[:3])
    if len(pred_classes) > 3:
        pred_text += "..."
    if true_classes:
        true_text = "\nTrue: " + ", ".join(true_classes[:3])
        if len(true_classes) > 3:
            true_text += "..."
        pred_text += true_text
    
    plt.figtext(0.5, 0.05, pred_text, ha='center', fontsize=10)
    plt.tight_layout()
    return fig

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot normalized confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    # cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    return plt

def generate_attention_map(model, image_tensor, device):
    """Generate attention heatmap using Integrated Gradients with proper baseline handling"""
    model.eval()
    integrated_gradients = IntegratedGradients(model)
    
    # Create proper baseline (black image with same shape)
    baseline = torch.zeros_like(image_tensor).to(device)
    
    # Compute attributions for each class and average
    num_classes = len(VOC_CLASSES)
    total_attributions = torch.zeros_like(image_tensor)
    
    for target_class in range(num_classes):
        attributions = integrated_gradients.attribute(
            image_tensor.unsqueeze(0),
            baselines=baseline.unsqueeze(0),
            target=target_class,
            n_steps=50
        )
        total_attributions += attributions.squeeze(0)
    
    # Average across classes and channels
    att_map = total_attributions.cpu().detach().numpy()
    att_map = np.mean(att_map, axis=0)  # Average across channels
    att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min())  # Normalize
    
    return att_map

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def visualize_predictions(model, dataloader, device, output_dir, num_samples=5):
    """Visualize model predictions on sample images and save each image separately"""
    model.eval()
    samples = []
    
    # Get random samples
    indices = np.random.choice(len(dataloader.dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        image, target = dataloader.dataset[idx]
        image = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image)
            probs = output.cpu().numpy()[0]

        image_tensor = image.squeeze(0).cpu().clone()
        image_tensor = denormalize(image_tensor, mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.255])
        image_tensor = torch.clamp(image_tensor, 0, 1)
        image = transforms.ToPILImage()(image_tensor)
        
        # Create figure for this sample
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        
        # Get top predictions
        pred_classes = []
        for j, prob in enumerate(probs):
            if prob > 0.5:
                pred_classes.append(f"{VOC_CLASSES[j]}: {prob:.2f}")
        
        # Get ground truth
        gt_classes = []
        for j, present in enumerate(target):
            if present == 1:
                gt_classes.append(VOC_CLASSES[j])
        
        # Add text to image
        plt.text(5, 15, f"Predicted: {', '.join(pred_classes)}", 
                color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        plt.text(5, 35, f"Ground Truth: {', '.join(gt_classes)}", 
                color='green', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
        # Save this sample separately
        sample_path = os.path.join(output_dir, f'sample_prediction_{i+1}.png')
        plt.savefig(sample_path, bbox_inches='tight')
        plt.close()
        
        # Add to samples list for potential later use
        samples.append((image, target, probs))
    
    return samples

def evaluate_model(model, dataloader, device):
    model.eval()
    criterion = nn.BCELoss()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).float().mean().item() * labels.size(0)
            total += labels.size(0)
            
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())
    
    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy

