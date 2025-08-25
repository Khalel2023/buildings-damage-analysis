import os
import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from dataset import BuildingSegmentationDataset
from unet import UNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--mask_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--accelerator', type=str, default='gpu')
    args = parser.parse_args()

    # Создаем датасеты
    train_dataset = BuildingSegmentationDataset(
        os.path.join(args.data_dir, 'train.txt'),
        args.image_dir,
        args.mask_dir,
        crop_size=args.crop_size,
        is_train=True
    )

    val_dataset = BuildingSegmentationDataset(
        os.path.join(args.data_dir, 'val.txt'),
        args.image_dir,
        args.mask_dir,
        crop_size=args.crop_size,
        is_train=False
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Модель
    model = UNet(lr=args.lr)

    # Коллбэки
    checkpoint_cb = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename='unet-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3
    )

    early_stop_cb = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )

    # Тренер
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        devices=args.devices,
        accelerator=args.accelerator,
        logger=TensorBoardLogger('tb_logs', name='unet'),
        callbacks=[checkpoint_cb, early_stop_cb],
        enable_progress_bar=True
    )



    # Обучение
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
