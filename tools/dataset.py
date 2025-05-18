import os
from torchvision.datasets import VOCDetection
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 定义数据转换
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 自定义数据集类（转换为多标签分类格式）
class VOCClassification(VOCDetection):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        
        # 提取多标签（20维二进制向量）
        labels = torch.zeros(len(VOC_CLASSES), dtype=torch.float32)
        for obj in target['annotation']['object']:
            cls = obj['name']
            if cls in VOC_CLASSES:
                labels[VOC_CLASSES.index(cls)] = 1.0
        
        return img, labels

