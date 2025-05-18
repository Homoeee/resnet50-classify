import os
from torchvision.datasets import VOCDetection
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from dataset import VOC_CLASSES, VOCClassification, transform
import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from utils import calculate_map, predict_image
import argparse
import time
from datetime import timedelta
import matplotlib.pyplot as plt


class VOCResNet(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        # 加载预训练ResNet
        self.backbone = models.resnet50(weights=weights)
        # 替换最后的全连接层
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, len(VOC_CLASSES))
        
    def forward(self, x):
        x = self.backbone(x)
        return torch.sigmoid(x)  # 多标签分类使用Sigmoid

def get_args():
    
    parser = argparse.ArgumentParser(description='Train a ResNet model on VOC dataset')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to VOC dataset')
    parser.add_argument('--year', type=str, default='2012', help='Year of the VOC dataset')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=150, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--save_dir', type=str, default='resnet50/checkpoints/2', help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to resume checkpoint')
    parser.add_argument('--weights', type=str, default='IMAGENET1K_V1', help='Pretrained weights for ResNet')
    parser.add_argument('--print_freq', type=int, default=10, help='Frequency of printing training status')

    return parser.parse_args()


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)

# 验证函数
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    total_correct = 0
    total_hamming_score = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # 将输出概率转换为二进制预测（阈值0.5）
            predicted = (outputs > 0.5).float()

            # 计算样本级别的准确率（完全匹配才算正确）
            exact_matches = (predicted == labels).all(dim=1).float().sum().item()
            total_correct += exact_matches

            # 计算汉明得分（Hamming Score，标签级别的准确率）
            hamming_score = (predicted == labels).float().mean().item()
            total_hamming_score += hamming_score * labels.size(0)

            total_samples += labels.size(0)

    avg_loss = running_loss / len(dataloader)
    exact_accuracy = total_correct / total_samples  # 完全匹配准确率
    hamming_accuracy = total_hamming_score / total_samples  # 汉明准确率

    return avg_loss, exact_accuracy, hamming_accuracy


def main():
    args = get_args()
    if os.path.exists(args.save_dir) is False:
        os.makedirs(args.save_dir)

    # 加载训练集和验证集
    train_dataset = VOCClassification(
        root=args.data_dir,
        year=args.year,
        image_set='train',
        download=False,
        transform=transform
    )
    val_dataset = VOCClassification(
        root=args.data_dir,
        year=args.year,
        image_set='val',
        download=False,
        transform=transform
    )

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    # 查看一个批次的数据
    images, labels = next(iter(train_loader))
    print("Batch shape:", images.shape)  # [32, 3, 224, 224]
    print("Labels shape:", labels.shape) # [32, 20]
    print("Sample label:", labels[20])    # 如 [0,1,0,...,1]（多标签）

    # 加载预训练 ResNet（移除顶层分类器）
    # weights = models.ResNet50_Weights.DEFAULT if args.weights == 'IMAGENET1K_V1' else None
    weights = None
    model = VOCResNet(weights=weights)

    # 移动到GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.BCELoss()  # 二分类交叉熵（多标签任务）
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 训练多个epoch
    num_epochs = args.num_epochs
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        epoch_start = time.time()

        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_total_acc, val_hamming_acc = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_start
        remaining_time = (args.num_epochs - epoch - 1) * epoch_time
        Map = calculate_map(model, val_loader, device)
        
        train_log = (f"Epoch [{epoch+1}/{num_epochs}]:"
            f"  Train Loss: {train_loss:.4f}"
            f"  Val Loss: {val_loss:.4f}, Val total Acc: {val_total_acc:.4f}"
            f"  Val hammign acc: {val_hamming_acc:.4f}"
            f'  Time: {timedelta(seconds=int(epoch_time))}'
            f'  ETA: {timedelta(seconds=int(remaining_time))}')
        print(train_log)
        # 将日志写入文件
        with open(os.path.join(args.save_dir, 'train_log.txt'), 'a') as f:
            f.write(train_log + '\n')

        # 保存最佳模型
        if val_hamming_acc > best_val_acc:
            best_val_acc = val_hamming_acc
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_hamming_acc': val_hamming_acc,
            }
            torch.save(checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Best model saved with accuracy: {best_val_acc:.4f}")
            
    # 绘制损失曲线并保存为图片
    plt.figure(figsize=(10, 6))
    plt.plot(range(args.num_epochs), train_losses, label='Train Loss', marker='o')
    plt.plot(range(args.num_epochs), val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss and Val Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.save_dir, 'loss_curve.png'))  # 保存为图片
    plt.close()  # 关闭图形，释放内存
    print("Loss curve saved as loss_curve.png")
    

if __name__ == "__main__":
    main()