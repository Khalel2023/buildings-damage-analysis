import os
import json
import argparse
import logging
import numpy as np
import torch
import cv2
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from shapely import wkt
from shapely.geometry import Polygon
import albumentations as A
from albumentations.pytorch import ToTensorV2
from imantics import Polygons, Mask
import matplotlib.pyplot as plt
from skimage import morphology
import sys

# Добавление пути к модулям моделей
from unet import UNet
from model.model import ConvNeXtDamageModel

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Константы
DAMAGE_ENCODING = {
    0: 'no-damage',
    1: 'minor-damage',
    2: 'major-damage',
    3: 'destroyed'
}

DAMAGE_COLORS = {
    'no-damage': (50, 255, 50),  # Яркий зеленый
    'minor-damage': (255, 255, 0),  # Желтый
    'major-damage': (0, 0, 255),  # Синий
    'destroyed': (255, 0, 0)  # Красный
}


class BuildingDamagePipeline:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._validate_paths()

        self.output_dir = Path(args.output_dir)
        self.temp_dir = self.output_dir / 'temp'
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _validate_paths(self):
        required = {
            'Pre-disaster image': self.args.pre_image,
            'Post-disaster image': self.args.post_image,
            'Segmentation weights': self.args.seg_weights,
            'Classification weights': self.args.cls_weights
        }
        for name, path in required.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"{name} not found at: {path}")

    def _load_model(self, model_class, weights_path):
        """Загрузка модели с обработкой различных форматов весов"""
        try:
            if model_class.__name__ == 'UNet':
                model = model_class.load_from_checkpoint(weights_path, map_location=self.device)
            else:
                model = model_class()
                state_dict = torch.load(weights_path, map_location=self.device)
                if 'state_dict' in state_dict:  # Для Lightning checkpoints
                    state_dict = state_dict['state_dict']
                model.load_state_dict(state_dict)
            return model.eval().to(self.device)
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def run_segmentation(self):
        """Улучшенная сегментация с предобработкой изображения"""
        logger.info("Running enhanced building segmentation...")

        # Улучшенная загрузка через PIL с конвертацией в numpy array
        try:
            image = np.array(Image.open(self.args.pre_image).convert("RGB"))
            if image is None:
                raise ValueError("Failed to load pre-disaster image")
        except Exception as e:
            logger.error(f"Error loading pre-disaster image: {str(e)}")
            raise

        image = cv2.GaussianBlur(image, (3, 3), 0)  # Легкое размытие для уменьшения шума

        # Улучшенные аугментации
        transform = A.Compose([
            A.CLAHE(clip_limit=2.0, p=0.5),  # Улучшение контраста
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        model = self._load_model(UNet, self.args.seg_weights)

        with torch.no_grad():
            input_tensor = transform(image=image)["image"].unsqueeze(0).to(self.device)
            output = model(input_tensor)
            prob_map = torch.sigmoid(output).squeeze().cpu().numpy()

            # Адаптивный порог
            if self.args.adaptive_threshold:
                threshold = max(0.3, np.mean(prob_map) * 1.5)
                mask = (prob_map > threshold).astype(np.uint8) * 255
            else:
                mask = (prob_map > self.args.seg_threshold).astype(np.uint8) * 255

        mask = self._postprocess_mask(mask)

        # Визуализация промежуточных результатов
        if self.args.debug:
            debug_dir = self.temp_dir / "debug"
            debug_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(debug_dir / "prob_map.png"), (prob_map * 255).astype(np.uint8))
            cv2.imwrite(str(debug_dir / "raw_mask.png"), mask)

        polygons = self._mask_to_polygons(mask, prob_map)

        seg_json = {
            "features": {"xy": polygons},
            "metadata": {
                "original_width": image.shape[1],
                "original_height": image.shape[0],
                "width": image.shape[1],
                "height": image.shape[0]
            }
        }

        output_path = self.temp_dir / "segmentation.json"
        with open(output_path, 'w') as f:
            json.dump(seg_json, f, indent=2)

        return output_path

    def _mask_to_polygons(self, mask, prob_map):
        """Конвертация маски в полигоны с улучшенной обработкой"""
        polygons = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            if len(contour) >= 3:  # Минимум 3 точки для полигона
                # Упрощение контура
                epsilon = 0.002 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                polygon = Polygon(approx.squeeze())
                if polygon.is_valid and not polygon.is_empty:
                    coords = list(polygon.exterior.coords)
                    if len(coords) >= 4:
                        # Вычисление уверенности по области полигона
                        x, y = approx.squeeze().T
                        confidence = np.mean(prob_map[y, x])

                        polygons.append({
                            "wkt": f"POLYGON (({','.join(f'{x} {y}' for x, y in coords)}))",
                            "properties": {
                                "feature_type": "building",
                                "uid": f"bld_{len(polygons)}",
                                "confidence": float(confidence)
                            }
                        })
        return polygons

    def _postprocess_mask(self, mask, min_area=100):
        """Улучшенная постобработка маски"""
        # 1. Морфологические операции
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 2. Удаление артефактов
        cleaned = morphology.remove_small_objects(mask.astype(bool), min_size=min_area)
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=min_area * 2)

        # 3. Сглаживание границ
        cleaned = cleaned.astype(np.uint8) * 255
        cleaned = cv2.GaussianBlur(cleaned, (3, 3), 0)
        _, cleaned = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)

        return cleaned

    def extract_buildings(self, seg_json_path):
        """Извлечение зданий с улучшенным ROI"""
        logger.info("Extracting building crops with enhanced ROI...")

        with open(seg_json_path) as f:
            seg_data = json.load(f)

        # Улучшенная загрузка post-изображения через PIL
        try:
            post_image = np.array(Image.open(self.args.post_image).convert("RGB"))
            if post_image is None:
                raise ValueError("Failed to load post-disaster image")
        except Exception as e:
            logger.error(f"Error loading post-disaster image: {str(e)}")
            raise

        crops_dir = self.temp_dir / "building_crops"
        crops_dir.mkdir(exist_ok=True)

        building_data = []
        for feature in tqdm(seg_data["features"]["xy"], desc="Processing buildings"):
            try:
                uid = feature["properties"]["uid"]
                polygon = wkt.loads(feature["wkt"])
                poly_pts = np.array(list(polygon.exterior.coords), dtype=np.int32)

                # Улучшенное вычисление ROI
                x, y = poly_pts[:, 0], poly_pts[:, 1]
                xmin, xmax = np.min(x), np.max(x)
                ymin, ymax = np.min(y), np.max(y)

                # Автоматический расчет padding
                width, height = xmax - xmin, ymax - ymin
                padding = int(max(width, height) * 0.15)  # 15% от максимального размера

                xmin = max(xmin - padding, 0)
                xmax = min(xmax + padding, post_image.shape[1])
                ymin = max(ymin - padding, 0)
                ymax = min(ymax + padding, post_image.shape[0])

                crop = post_image[ymin:ymax, xmin:xmax]
                crop_path = crops_dir / f"{uid}.png"

                # Сохраняем через PIL для надежности
                Image.fromarray(crop).save(str(crop_path))

                building_data.append({
                    "uid": uid,
                    "path": str(crop_path),
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax
                })

            except Exception as e:
                logger.warning(f"Failed to process building {uid}: {str(e)}")

        csv_path = self.temp_dir / "buildings.csv"
        pd.DataFrame(building_data).to_csv(csv_path, index=False)
        return csv_path, crops_dir

    def classify_damage(self, csv_path, crops_dir):
        """Классификация повреждений с улучшенными аугментациями"""
        logger.info("Classifying damage with enhanced augmentations...")

        model = self._load_model(ConvNeXtDamageModel, self.args.cls_weights)

        class BuildingDataset(torch.utils.data.Dataset):
            def __init__(self, df, transform=None):
                self.df = df
                self.transform = transform or A.Compose([
                    A.RandomRotate90(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Resize(224, 224),
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])

            def __len__(self):
                return len(self.df)

            def __getitem__(self, idx):
                img_path = self.df.iloc[idx]["path"]
                try:
                    image = np.array(Image.open(img_path).convert("RGB"))
                    if self.transform:
                        image = self.transform(image=image)["image"]
                    return image, self.df.iloc[idx]["uid"]
                except Exception as e:
                    logger.error(f"Error loading image {img_path}: {str(e)}")
                    raise

        df = pd.read_csv(csv_path)
        dataset = BuildingDataset(df)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)

        results = {}
        with torch.no_grad():
            for images, uids in tqdm(loader, desc="Classifying"):
                try:
                    outputs = model(images.to(self.device))
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    for uid, pred, prob in zip(uids, preds, probs):
                        results[uid] = {
                            "damage": DAMAGE_ENCODING[pred.item()],
                            "confidence": float(torch.max(prob))
                        }
                except Exception as e:
                    logger.error(f"Error during classification: {str(e)}")
                    continue

        output_path = self.temp_dir / "damage_classification.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        return output_path

    def combine_results(self, seg_json_path, damage_json_path):
        """Объединение результатов с дополнительной информацией"""
        logger.info("Merging enhanced results...")

        with open(seg_json_path) as f:
            seg_data = json.load(f)
        with open(damage_json_path) as f:
            damage_data = json.load(f)

        for feature in seg_data["features"]["xy"]:
            uid = feature["properties"]["uid"]
            if uid in damage_data:
                feature["properties"].update(damage_data[uid])

        output_path = self.output_dir / "final_results.json"
        with open(output_path, 'w') as f:
            json.dump(seg_data, f, indent=2)

        return output_path

    def visualize_results(self, final_json_path):
        """Визуализация с гарантированно правильными цветами масок"""
        logger.info("Generating visualization with exact mask colors...")

        try:
            # 1. Загрузка JSON данных
            with open(final_json_path) as f:
                data = json.load(f)

            # 2. Надежная загрузка изображения
            try:
                # Способ 1: через PIL (более надежный)
                pil_image = Image.open(self.args.post_image).convert("RGB")
                image = np.array(pil_image)
                if image is None:
                    # Способ 2: через OpenCV если PIL не сработал
                    image = cv2.imread(str(self.args.post_image))
                    if image is None:
                        raise ValueError("Не удалось загрузить изображение")
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.error(f"Ошибка загрузки изображения: {str(e)}")
                raise

            # 3. Подготовка слоев
            overlay = image.copy()
            outline = np.zeros_like(image)

            # 4. Применение масок с ТОЧНО вашими цветами
            for feature in data["features"]["xy"]:
                damage = feature["properties"].get("damage", "no-damage")
                color = DAMAGE_COLORS[damage]  # Используем ваши оригинальные цвета

                polygon = wkt.loads(feature["wkt"])
                pts = np.array(list(polygon.exterior.coords), dtype=np.int32)

                # Заливка (ваши точные цвета)
                cv2.fillPoly(overlay, [pts], color)
                # Контур (ваши точные цвета и толщина 2px)
                cv2.polylines(outline, [pts], isClosed=True, color=color, thickness=2)

            # 5. Композиция (ваши оригинальные коэффициенты)
            result = cv2.addWeighted(image, 0.5, overlay, 0.3, 0)
            result = cv2.addWeighted(result, 0.8, outline, 0.2, 0)

            # 6. Надежное сохранение
            output_path = Path(self.output_dir) / "visualization.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Сохраняем через PIL для точной цветопередачи
            Image.fromarray(result).save(output_path)

            # 7. Сохранение легенды
            self._save_legend()

            return output_path

        except Exception as e:
            logger.error(f"Ошибка визуализации: {str(e)}", exc_info=True)
            raise RuntimeError(f"Ошибка визуализации: {str(e)}") from e
    def _save_legend(self):
        """Сохранение легенды цветов"""
        legend = np.zeros((200, 300, 3), dtype=np.uint8)
        y = 30
        for i, (damage, color) in enumerate(DAMAGE_COLORS.items()):
            cv2.putText(legend, damage, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.rectangle(legend, (10, y - 15), (40, y + 5), color, -1)
            y += 30

        cv2.imwrite(str(self.output_dir / "legend.png"), cv2.cvtColor(legend, cv2.COLOR_RGB2BGR))

    def run(self):
        """Основной цикл выполнения с улучшенной обработкой ошибок"""
        try:
            seg_json = self.run_segmentation()
            buildings_csv, crops_dir = self.extract_buildings(seg_json)
            damage_json = self.classify_damage(buildings_csv, crops_dir)
            final_json = self.combine_results(seg_json, damage_json)
            vis_path = self.visualize_results(final_json)

            logger.info(f"Enhanced pipeline completed! Results saved to: {self.output_dir}")
            return {
                'segmentation': seg_json,
                'building_crops': buildings_csv,
                'damage_classification': damage_json,
                'final_results': final_json,
                'visualization': vis_path
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise


def main():
    parser = argparse.ArgumentParser(description='Enhanced Building Damage Assessment Pipeline')
    parser.add_argument('--pre_image', required=True, help='Pre-disaster image path')
    parser.add_argument('--post_image', required=True, help='Post-disaster image path')
    parser.add_argument('--seg_weights', required=True, help='Segmentation model weights path')
    parser.add_argument('--cls_weights', required=True, help='Classification model weights path')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--seg_threshold', type=float, default=0.5, help='Segmentation threshold')
    parser.add_argument('--adaptive_threshold', action='store_true', help='Use adaptive thresholding')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for classification')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with intermediate outputs')
    args = parser.parse_args()

    try:
        pipeline = BuildingDamagePipeline(args)
        results = pipeline.run()
        logger.info(f"Success! Final results: {results['final_results']}")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
