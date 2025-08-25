import os
import shutil
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys
import cv2


def validate_mask(mask_path):
    if not os.path.exists(mask_path):
        return False
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return mask is not None and np.any(mask > 0)


def find_mask_variants(base_name):
    return [
        f"damage_{base_name}",
        base_name.replace('_post_disaster.png', '_post_disaster_mask.png'),
        base_name.replace('_post_disaster.png', '_damage.png'),
        base_name.replace('.png', 'post_mask.png'),
        'post_mask_' + base_name,
        base_name
    ]


def process_disaster(disaster_path, output_dir):
    images_path = os.path.join(disaster_path, "images")
    masks_path = os.path.join(disaster_path, "labels")

    if not os.path.exists(images_path):
        print(f"Директория с изображениями не найдена: {images_path}")
        return []

    suffix = '_post_disaster.png'
    images = [f for f in os.listdir(images_path) if f.endswith(suffix)]
    valid_pairs = []

    for img in tqdm(images, desc='Обработка post-disaster'):
        img_path = os.path.join(images_path, img)

        for mask_name in find_mask_variants(img):
            mask_path = os.path.join(masks_path, mask_name)
            if os.path.exists(mask_path) and validate_mask(mask_path):
                new_mask_name = f"post_mask_{img}"
                shutil.copy2(img_path, os.path.join(output_dir, "images", img))
                shutil.copy2(mask_path, os.path.join(output_dir, "labels", new_mask_name))
                valid_pairs.append(img)
                break
        else:
            print(f"\nПредупреждение: Не найдена валидная маска для {img}")

    return valid_pairs


def main(input_dir, output_dir, split_ratio=0.8):
    os.makedirs(output_dir, exist_ok=True)
    for subdir in ["images", "labels", "dataSet"]:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    if os.path.basename(input_dir) in ['train', 'test', 'val']:
        disasters = [input_dir]
    else:
        disasters = [d for d in os.listdir(input_dir)
                     if os.path.isdir(os.path.join(input_dir, d))]

    all_images = []
    for disaster in disasters:
        disaster_path = os.path.join(input_dir, disaster)
        all_images.extend(process_disaster(disaster_path, output_dir))

    if not all_images:
        print("\nОшибка: Не найдено подходящих изображений")
        sys.exit(1)

    unique_disasters = list({img.split('_')[0] for img in all_images})
    train_disasters, val_disasters = train_test_split(
        unique_disasters, test_size=1 - split_ratio, random_state=42
    )

    train_files = [img for img in all_images if img.split('_')[0] in train_disasters]
    val_files = [img for img in all_images if img.split('_')[0] in val_disasters]

    with open(os.path.join(output_dir, "dataSet", "train.txt"), 'w') as f:
        f.write('\n'.join(train_files))
    with open(os.path.join(output_dir, "dataSet", "val.txt"), 'w') as f:
        f.write('\n'.join(val_files))

    print("\nРезультат:")
    print(f"Всего post-disaster изображений: {len(all_images)}")
    print(f"Тренировочные: {len(train_files)}")
    print(f"Валидационные: {len(val_files)}")
    print(f"Выходная директория: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Обработка только POST-disaster изображений xBD')
    parser.add_argument('--input', required=True, help='Директория с данными xBD')
    parser.add_argument('--output', default='spacenet_post_gt', help='Выходной каталог')
    parser.add_argument('--split', type=float, default=0.8, help='Соотношение train/val')
    args = parser.parse_args()

    print("\n" + "="*50)
    print("Обработка POST-disaster изображений".center(50))
    print("="*50)
    main(
        input_dir=args.input,
        output_dir=args.output,
        split_ratio=args.split
    )
