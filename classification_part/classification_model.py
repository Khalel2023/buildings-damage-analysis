import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from PIL import Image
import numpy as np
import argparse
from model import ConvNeXtDamageModel, FocalLoss


class DamageDataset(Dataset):
    def __init__(self, csv_path, img_dir, augment=False):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.augment = augment

        # Базовые преобразования
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Аугментации
        self.augment_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['uuid']
        img_path = os.path.join(self.img_dir, f"{img_name}.png")
        image = Image.open(img_path).convert('RGB')
        label = self.df.iloc[idx]['labels']

        transform = self.augment_transform if self.augment else self.base_transform
        return transform(image), label


def get_class_weights(df, beta=0.999):
    """Вычисление весов классов с эффективным числом samples"""
    class_counts = df['labels'].value_counts().sort_index()
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / weights.sum() * len(class_counts)
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    if scheduler:
        scheduler.step()

    train_loss = running_loss / len(loader)
    train_acc = accuracy_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds, average='weighted')

    return train_loss, train_acc, train_f1


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(loader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds)

    return val_loss, val_acc, val_f1, report


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Загрузка данных
    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    # Вычисление весов классов
    class_weights = get_class_weights(train_df).to(device)
    print(f"Class weights: {class_weights.cpu().numpy()}")

    # Датасеты
    train_dataset = DamageDataset(args.train_csv, args.img_dir, augment=True)
    val_dataset = DamageDataset(args.val_csv, args.img_dir)

    # Проверка формы данных
    sample_img, sample_label = train_dataset[0]
    print(f"Sample image shape: {sample_img.shape}")

    # WeightedRandomSampler для балансировки классов
    sample_weights = class_weights[train_df['labels'].values]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Инициализация модели
    model = ConvNeXtDamageModel().to(device)
    print(f"Model architecture:\n{model}")

    # Функция потерь и оптимизатор
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Обучение
    best_f1 = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Фаза обучения
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, scheduler)

        # Фаза валидации
        val_loss, val_acc, val_f1, report = validate(
            model, val_loader, criterion, device)

        # Логирование
        print(f"\nTrain Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        print("\nClassification Report:\n", report)

        # Сохранение лучшей модели
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print(f"Saved best model with F1: {val_f1:.4f}")

    # Сохранение финальной модели
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Damage Classification Training')
    parser.add_argument('--train_csv', required=True, help='Path to training CSV file')
    parser.add_argument('--val_csv', required=True, help='Path to validation CSV file')
    parser.add_argument('--img_dir', required=True, help='Directory containing images')
    parser.add_argument('--output_dir', required=True, help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    args = parser.parse_args()

    # Создание выходной директории
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    main(args)
