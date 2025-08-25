import os
from PIL import Image
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from shapely.geometry import Polygon
from sklearn.model_selection import train_test_split
import logging
from concurrent.futures import ProcessPoolExecutor
from shapely import wkt
from skimage.filters import laplace
from skimage.transform import resize
from sklearn.utils import shuffle
from functools import partial
import argparse

# Настройка логгирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Конфигурация
DAMAGE_ENCODING = {
    'no-damage': 0,
    'minor-damage': 1,
    'major-damage': 2,
    'destroyed': 3
}
MIN_POLYGON_AREA = 100
TARGET_SIZE = (256, 256)
BLUR_THRESHOLD = 100
BUFFER_PCT = 0.2
MAX_SAMPLES_PER_CLASS = {
    0: 8000,  # no-damage
    1: 4000,  # minor-damage
    2: 4000,  # major-damage
    3: 4000  # destroyed
}


def is_blurry(image):
    """Проверка на размытость с использованием лапласиана"""
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    return laplace(image).var() < BLUR_THRESHOLD


def normalize_image(img):
    """Нормализация изображения"""
    img = img.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-7)
    return img


def process_building(img_array, polygon, damage_type, uuid, output_dir, is_pre=False):
    """Обработка одного здания"""
    x, y = polygon.exterior.xy
    xmin, xmax = int(min(x)), int(max(x))
    ymin, ymax = int(min(y)), int(max(y))

    x_buffer = int((xmax - xmin) * BUFFER_PCT)
    y_buffer = int((ymax - ymin) * BUFFER_PCT)

    xmin = max(0, xmin - x_buffer)
    xmax = min(img_array.shape[1], xmax + x_buffer)
    ymin = max(0, ymin - y_buffer)
    ymax = min(img_array.shape[0], ymax + y_buffer)

    cropped = img_array[ymin:ymax, xmin:xmax]
    if min(cropped.shape[:2]) < 50:
        return None

    if len(cropped.shape) == 2:
        cropped = np.stack([cropped] * 3, axis=-1)

    cropped = resize(cropped, TARGET_SIZE, preserve_range=True, anti_aliasing=True)
    cropped = normalize_image(cropped)

    suffix = "_pre" if is_pre else "_post"
    output_path = os.path.join(output_dir, f"{uuid}{suffix}.png")

    # Сохранение как PNG вместо TIFF
    img_to_save = (cropped * 255).astype(np.uint8)
    Image.fromarray(img_to_save).save(output_path)

    return {
        'uuid': f"{uuid}{suffix}",
        'labels': 0 if is_pre else DAMAGE_ENCODING[damage_type],
        'disaster': os.path.basename(os.path.dirname(output_dir))
    }


def process_image_pair(post_img_path, label_path, output_dir):
    """Обработка пары pre/post изображений"""
    try:
        # Чтение PNG вместо TIFF
        post_img = np.array(Image.open(post_img_path))
        if is_blurry(post_img):
            return None

        with open(label_path) as f:
            label_data = json.load(f)

        pre_img_path = post_img_path.replace("_post_", "_pre_")
        pre_img = np.array(Image.open(pre_img_path)) if os.path.exists(pre_img_path) else None

        results = []
        buildings_processed = set()

        for feat in label_data['features']['xy']:
            damage_type = feat.get('properties', {}).get('subtype', 'no-damage')
            if damage_type not in DAMAGE_ENCODING:
                continue

            polygon = wkt.loads(feat['wkt'])
            if polygon.area < MIN_POLYGON_AREA:
                continue

            uuid = feat['properties']['uid']
            if uuid in buildings_processed:
                continue
            buildings_processed.add(uuid)

            post_result = process_building(post_img, polygon, damage_type, uuid, output_dir)
            if post_result:
                results.append(post_result)

                if pre_img is not None:
                    pre_result = process_building(pre_img, polygon, damage_type, uuid, output_dir, is_pre=True)
                    if pre_result:
                        results.append(pre_result)

        return results

    except Exception as e:
        logger.error(f"Error processing {post_img_path}: {str(e)}")
        return None


def balance_data(df):
    """Балансировка данных"""
    balanced = []

    for class_id in [1, 2, 3]:
        class_df = df[df['labels'] == class_id]
        if len(class_df) > MAX_SAMPLES_PER_CLASS[class_id]:
            class_df = shuffle(class_df).iloc[:MAX_SAMPLES_PER_CLASS[class_id]]
        balanced.append(class_df)

    max_damaged = sum(len(c) for c in balanced)
    no_damage = df[df['labels'] == 0]
    if len(no_damage) > max_damaged:
        no_damage = shuffle(no_damage).iloc[:max_damaged]
    balanced.append(no_damage)

    return pd.concat(balanced)


def prepare_directories(output_dir, output_csv_dir):
    """Создание выходных директорий"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_csv_dir, exist_ok=True)
    logger.info(f"Output images directory: {output_dir}")
    logger.info(f"Output CSV directory: {output_csv_dir}")


def collect_tasks(input_dir):
    """Сбор задач для обработки"""
    tasks = []
    for disaster in os.listdir(input_dir):
        disaster_path = os.path.join(input_dir, disaster)
        if not os.path.isdir(disaster_path):
            continue

        images_dir = os.path.join(disaster_path, "images")
        labels_dir = os.path.join(disaster_path, "labels")

        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            logger.warning(f"Missing images or labels in {disaster_path}")
            continue

        for img_file in os.listdir(images_dir):
            if not img_file.endswith('.png') or '_post_' not in img_file:
                continue

            img_path = os.path.join(images_dir, img_file)
            label_path = os.path.join(labels_dir, img_file.replace('.png', '.json'))
            if os.path.exists(label_path):
                tasks.append((img_path, label_path))

    if not tasks:
        logger.error("No valid image-label pairs found!")
        return None
    return tasks


def process_single_task(task_args, output_dir):
    """Обрабатывает одну задачу (пару изображение-метка)"""
    img_path, label_path = task_args
    return process_image_pair(img_path, label_path, output_dir)


def process_and_save_data(tasks, output_dir, output_csv_dir, val_split):
    """Обработка данных и сохранение результатов"""
    all_results = []

    # Используем partial для фиксации output_dir
    process_func = partial(process_single_task, output_dir=output_dir)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Правильно передаем tasks в executor.map
        for result in tqdm(executor.map(process_func, tasks),
                           total=len(tasks), desc="Processing images"):
            if result:
                all_results.extend(result)

    if not all_results:
        logger.error("No valid results after processing!")
        return False

    df = pd.DataFrame(all_results)
    logger.info(f"Raw class distribution:\n{df['labels'].value_counts()}")

    df = balance_data(df)
    logger.info(f"Balanced class distribution:\n{df['labels'].value_counts()}")

    train_df, val_df = train_test_split(
        df,
        test_size=val_split,
        stratify=df['labels'],
        random_state=42
    )

    train_csv_path = os.path.join(output_csv_dir, "train.csv")
    val_csv_path = os.path.join(output_csv_dir, "test.csv")
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)

    logger.info(f"Processing complete. Results saved to:")
    logger.info(f"Images: {output_dir} ({len(df)} images)")
    logger.info(f"Train CSV: {train_csv_path} ({len(train_df)} samples)")
    logger.info(f"Test CSV: {val_csv_path} ({len(val_df)} samples)")
    return True


def main():
    parser = argparse.ArgumentParser(description='Process xBD dataset')
    parser.add_argument('--input_dir', required=True, help='Input data directory')
    parser.add_argument('--output_dir', required=True, help='Output images directory')
    parser.add_argument('--output_csv_dir', required=True, help='Output CSV directory')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    args = parser.parse_args()

    prepare_directories(args.output_dir, args.output_csv_dir)
    tasks = collect_tasks(args.input_dir)
    if tasks:
        process_and_save_data(tasks, args.output_dir, args.output_csv_dir, args.val_split)


if __name__ == "__main__":
    main()
