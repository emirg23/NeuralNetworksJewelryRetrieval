import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ArcFaceModel(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.embedding = nn.Linear(512, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)

        x = F.normalize(x, p=2, dim=1)
        return x
