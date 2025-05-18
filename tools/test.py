import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import VOC_CLASSES, VOCClassification
from torchvision import models
import torch.nn as nn
import argparse
from utils import predict_image, plot_attention, generate_attention_map, visualize_predictions
from utils import plot_confusion_matrix, evaluate_model
from PIL import Image
import json
from sklearn.metrics import classification_report

class VOCResNet(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.backbone = models.resnet50(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, len(VOC_CLASSES))
        
    def forward(self, x):
        x = self.backbone(x)
        return torch.sigmoid(x)

def get_args():
    parser = argparse.ArgumentParser(description='Test a ResNet model on VOC dataset')
    parser.add_argument('--model_path', type=str, default='resnet50/checkpoints/2/best_model.pth', help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to VOC dataset')
    parser.add_argument('--output_dir', type=str, default='resnet50/results/3', help='Directory to save test results')
    parser.add_argument('--year', type=str, default='2012', help='Year of the VOC dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--test_image', type=str, default=None, help='Path to single test image')
    parser.add_argument('--save_predictions', default=True, help='Save predictions to JSON file')
    parser.add_argument('--visualize', default=True, help='Generate visualizations')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to visualize')
    return parser.parse_args()

def make_serializable(obj):
    """
    Recursively convert numpy objects to native Python types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return make_serializable(obj.tolist())
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load model
    model = VOCResNet()
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
    # Transform for test images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if args.test_image:
        # Single image inference with visualization
        image_path = args.test_image
        classes, probs = predict_image(image_path, model, device, transform)
        
        print("\nSingle Image Prediction Results:")
        print(f"Image: {image_path}")
        print("Predicted Classes with Probabilities:")
        for cls, prob in zip(classes, probs):
            print(f"  {cls}: {prob:.4f}")
        
        # Generate attention visualization
        image_tensor = transform(Image.open(image_path)).to(device)
        att_map = generate_attention_map(model, image_tensor, device)
        image = Image.open(image_path).resize((224, 224))
        fig = plot_attention(image, att_map, classes)
        fig.savefig(os.path.join(args.output_dir, 'single_image_attention.png'))
        plt.close(fig)
        
        # Save results
        result = {
            'image': image_path,
            'predictions': {cls: float(prob) for cls, prob in zip(VOC_CLASSES, probs)},
            'attention_map': att_map.tolist()
        }
        with open(os.path.join(args.output_dir, 'single_prediction.json'), 'w') as f:
            json.dump(result, f, indent=4)
    else:
        # Full test set evaluation
        test_dataset = VOCClassification(
            root=args.data_dir,
            year=args.year,
            image_set='test' if args.year == '2007' else 'val',
            download=False,
            transform=transform
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        
        test_loss, test_acc = evaluate_model(model, test_loader, device)
        
        print("\nTest Set Evaluation Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        
        if args.visualize:
            print("\nGenerating visualizations...")
            # Visualize sample predictions
            visualize_predictions(model, test_loader, device, args.output_dir, args.num_samples)
            
            # Generate confusion matrix (for most frequent class)
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    outputs = model(images)
                    preds = (outputs > 0.5).float().cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())
            
            # For visualization, pick the most frequent class
            class_counts = np.sum(all_labels, axis=0)
            print("Class Counts:", class_counts)
            dominant_class = np.argmax(class_counts)
            y_true = [label[dominant_class] for label in all_labels]
            y_pred = [pred[dominant_class] for pred in all_preds]
            
            cm_plot = plot_confusion_matrix(y_true, y_pred, ['negetive', VOC_CLASSES[dominant_class]])
            cm_plot.savefig(os.path.join(args.output_dir, 'confusion_matrix.png'))
            cm_plot.close()
            
            # Generate class-wise performance plot
            class_acc = []
            for i in range(len(VOC_CLASSES)):
                correct = np.sum([(p[i] == l[i]) for p, l in zip(all_preds, all_labels)])
                total = len(all_labels)
                class_acc.append(correct / total)
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(VOC_CLASSES, class_acc)
            # 在每个柱上添加文本
            for bar in bars:
                yval = bar.get_height()  # 获取柱的高度（即准确率）
                plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
            plt.xticks(rotation=45)
            plt.title('Class-wise Accuracy')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'class_accuracy.png'))
            plt.close()
        
        if args.save_predictions:
            # 构建结果字典
            results = {
                'metrics': {
                    'loss': float(test_loss),
                    'accuracy': float(test_acc),
                },
                'class_metrics': None,
                'predictions': None
            }

            # 添加 class_metrics（如果有）
            if args.visualize:
                class_metrics = classification_report(
                    np.array(all_labels),
                    np.array(all_preds) > 0.5,
                    target_names=VOC_CLASSES,
                    output_dict=True
                )
                results['class_metrics'] = make_serializable(class_metrics)
                results['predictions'] = make_serializable(all_preds)

            # 保存到 JSON 文件
            with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
                json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()