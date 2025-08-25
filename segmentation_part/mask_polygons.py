import os
import json
import numpy as np
import cv2
from shapely import wkt
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DisasterMaskCreator:
    def __init__(self, mode='pre'):
        self.mask_prefix = "mask_"
        self.mode = mode
        self.required_suffix = "_pre_disaster" if mode == 'pre' else "_post_disaster"

    def process_root_directory(self, root_dir):
        """Обрабатывает директорию с катастрофами или конкретную катастрофу"""
        logger.info(f"Начинаем обработку: {root_dir} (режим: {self.mode})")

        # Проверяем, является ли это папкой катастрофы (содержит images/labels)
        if os.path.exists(os.path.join(root_dir, "images")):
            self._process_single_disaster(root_dir)
        else:
            # Ищем подпапки с катастрофами
            disaster_folders = [d for d in os.listdir(root_dir)
                                if os.path.isdir(os.path.join(root_dir, d))]

            if not disaster_folders:
                logger.error(f"Не найдено папок катастроф в {root_dir}")
                return

            for disaster in tqdm(disaster_folders, desc='Обработка катастроф'):
                disaster_path = os.path.join(root_dir, disaster)
                if os.path.exists(os.path.join(disaster_path, "images")):
                    self._process_single_disaster(disaster_path)

    def _process_single_disaster(self, disaster_path):
        """Обрабатывает одну катастрофу"""
        images_dir = os.path.join(disaster_path, "images")
        labels_dir = os.path.join(disaster_path, "labels")
        masks_dir = os.path.join(disaster_path, "masks")

        os.makedirs(masks_dir, exist_ok=True)

        images = [f for f in os.listdir(images_dir)
                  if f.endswith('.png') and self.required_suffix in f]

        if not images:
            logger.warning(f"Нет {self.mode}-disaster изображений в {images_dir}")
            return

        for img_file in tqdm(images, desc=f'Обработка {os.path.basename(disaster_path)}'):
            img_path = os.path.join(images_dir, img_file)
            json_file = img_file.replace('.png', '.json')
            json_path = os.path.join(labels_dir, json_file)
            mask_path = os.path.join(masks_dir, f"{self.mask_prefix}{img_file}")

            self._create_single_mask(img_path, json_path, mask_path)

    def _create_single_mask(self, img_path, json_path, mask_path):
        """Создает маску для одного изображения"""
        try:
            img = cv2.imread(img_path)
            if img is None:
                logger.error(f"Не удалось загрузить: {img_path}")
                return False

            height, width = img.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)

            if os.path.exists(json_path):
                with open(json_path) as f:
                    features = json.load(f)
                polygons = self._extract_polygons(features)
                for poly in polygons:
                    cv2.fillPoly(mask, [poly], 255)
            elif self.mode == 'post':
                # Улучшенная обработка для post-disaster
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) > 100:
                        cv2.drawContours(mask, [cnt], -1, 255, -1)

            cv2.imwrite(mask_path, mask)
            return True
        except Exception as e:
            logger.error(f"Ошибка обработки {img_path}: {str(e)}")
            return False

    def _extract_polygons(self, features):
        """Извлекает полигоны из GeoJSON"""
        polygons = []
        if not features.get('features', {}).get('xy'):
            return polygons

        for feat in features['features']['xy']:
            try:
                geom = wkt.loads(feat['wkt'])
                coords = np.array(list(geom.exterior.coords), dtype=np.int32)
                polygons.append(coords)
            except Exception:
                continue
        return polygons


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Создание масок для изображений катастроф")
    parser.add_argument('--input', required=True, help='Путь к корневой директории или папке катастрофы')
    parser.add_argument('--mode', choices=['pre', 'post'], default='pre',
                        help='Режим обработки: pre или post-disaster')
    parser.add_argument('--debug', action='store_true', help='Включить детальное логирование')

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    try:
        masker = DisasterMaskCreator(mode=args.mode)
        masker.process_root_directory(root_dir=args.input)
        logger.info("Обработка завершена успешно!")
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}")
        exit(1)