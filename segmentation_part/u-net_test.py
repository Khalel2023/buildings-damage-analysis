import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import BuildingSegmentationDataset
from unet import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2


def visualize_results(model, test_loader, device, save_dir='results', num_samples=5):
    """
    Визуализирует предсказания модели на тестовых данных
    Args:
        model: обученная модель
        test_loader: DataLoader с тестовыми данными
        device: устройство для вычислений (cuda/cpu)
        save_dir: директория для сохранения результатов
        num_samples: количество примеров для визуализации
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            if i >= num_samples:
                break

            images = images.to(device)
            masks = masks.to(device)

            # Получаем предсказания
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            # Преобразуем тензоры в numpy
            images = images.cpu().numpy()
            masks = masks.cpu().numpy()
            preds = preds.cpu().numpy()

            # Визуализируем результаты
            for j in range(images.shape[0]):
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))

                # Оригинальное изображение
                img = np.transpose(images[j], (1, 2, 0))
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Денормализация
                ax[0].imshow(np.clip(img, 0, 1))
                ax[0].set_title('Original Image')
                ax[0].axis('off')

                # Истинная маска
                ax[1].imshow(masks[j], cmap='gray')
                ax[1].set_title('Ground Truth')
                ax[1].axis('off')

                # Предсказанная маска
                ax[2].imshow(preds[j][0], cmap='gray')
                ax[2].set_title('Prediction')
                ax[2].axis('off')

                plt.savefig(os.path.join(save_dir, f'sample_{i}_{j}.png'))
                plt.close()


def main():
    # Параметры
    checkpoint_path = 'path/to/your/checkpoint.ckpt'  # Укажите путь к вашему чекпоинту
    data_dir = 'path/to/your/data'  # Директория с данными
    image_dir = 'path/to/post_disaster/images'  # Директория с изображениями
    mask_dir = 'path/to/post_disaster/masks'  # Директория с масками

    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Загружаем модель
    model = UNet.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.freeze()

    # Создаем тестовый датасет и loader
    test_dataset = BuildingSegmentationDataset(
        os.path.join(data_dir, 'test.txt'),  # Убедитесь, что у вас есть test.txt
        image_dir,
        mask_dir,
        crop_size=256,
        is_train=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4
    )

    # Визуализируем результаты
    visualize_results(model, test_loader, device)


if __name__ == '__main__':
    main()
