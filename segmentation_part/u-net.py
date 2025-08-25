import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
import pytorch_lightning as pl


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # Encoder
        self.inc = DoubleConv(3, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512)

        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)

        # Output (1 channel with sigmoid)
        self.outc = nn.Conv2d(64, 1, kernel_size=1)

        # Metrics
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = F.max_pool2d(x1, 2)
        x2 = self.down1(x2)
        x3 = F.max_pool2d(x2, 2)
        x3 = self.down2(x3)
        x4 = F.max_pool2d(x3, 2)
        x4 = self.down3(x4)

        # Decoder
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)

        return self.outc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(1)  # (B,1,H,W) -> (B,H,W)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        preds = torch.sigmoid(y_hat) > 0.5
        self.train_acc(preds, y)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        preds = torch.sigmoid(y_hat) > 0.5
        self.val_acc(preds, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
