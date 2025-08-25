import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class ConvNeXtDamageModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Загрузка предобученной ConvNeXt (tiny версия)
        self.backbone = models.convnext_tiny(pretrained=True)

        # Замена классификатора
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Добавляем слой пулинга
            nn.Flatten(),
            nn.LayerNorm(in_features, eps=1e-6),
            nn.Linear(in_features, num_classes)
        )

        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.backbone(x)
