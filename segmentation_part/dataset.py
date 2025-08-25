import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BuildingSegmentationDataset(Dataset):
    def __init__(self,
                 data_list_path: str,
                 images_dir: str,
                 masks_dir: str,
                 crop_size: int = 256,
                 is_train: bool = True):
        """
        Args:
            data_list_path: Путь к файлу train.txt/val.txt
            images_dir: Директория с изображениями
            masks_dir: Директория с масками (должны иметь префикс 'mask_')
            crop_size: Размер обрезки
            is_train: Режим обучения (True) или валидации (False)
        """
        with open(data_list_path, 'r') as f:
            self.image_names = [line.strip() for line in f if line.strip()]

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.crop_size = crop_size
        self.is_train = is_train

        # Базовые трансформации
        self.base_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        # Аугментации только для тренировочного набора
        self.train_transform = A.Compose([
            A.RandomCrop(crop_size, crop_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5)
        ])

        # Трансформации для валидации
        self.val_transform = A.Compose([
            A.CenterCrop(crop_size, crop_size)
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]

        # 1. Загрузка изображения
        image_path = os.path.join(self.images_dir, image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = np.array(Image.open(image_path).convert('RGB'))

        # 2. Загрузка маски с префиксом 'mask_'
        mask_name = f"post_mask_{image_name}"
        mask_path = os.path.join(self.masks_dir, mask_name)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        mask = np.array(Image.open(mask_path).convert('L'))
        mask = (mask > 0).astype(np.float32)  # Конвертируем в 0 и 1

        # 3. Применяем трансформации
        if self.is_train:
            transformed = self.train_transform(image=image, mask=mask)
        else:
            transformed = self.val_transform(image=image, mask=mask)

        image = transformed['image']
        mask = transformed['mask']

        # 4. Базовые трансформации (нормализация и ToTensor)
        transformed = self.base_transform(image=image, mask=mask)

        # Возвращаем изображение и маску (удаляем channel dim для маски)
        return transformed['image'], transformed['mask'].squeeze(0)

    def __repr__(self):
        return (f"BuildingSegmentationDataset(len={len(self)}, "
                f"images_dir='{self.images_dir}', "
                f"masks_dir='{self.masks_dir}', "
                f"crop_size={self.crop_size}, "
