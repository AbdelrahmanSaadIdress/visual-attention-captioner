import torch.nn as nn
from torchvision import models


class ResNet50(nn.Module):
    """ResNet50 backbone stripped of avgpool/fc, kept as a spatial feature map.

    Output is (B, 49, 2048): 49 = 7x7 spatial locations, 2048 channels,
    ready to feed into the attention module. All parameters start frozen;
    the training script selectively unfreezes later layers (e.g. layer3/4)
    once cross-entropy warmup is done.
    """

    def __init__(self):
        super().__init__()
        self.ResNet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(self.ResNet50.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        for param in self.ResNet50.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        B, C, H, W = x.size()
        x = x.view(B, C, -1)     # (B, 2048, 49)
        x = x.permute(0, 2, 1)   # (B, 49, 2048) -- 49 spatial locations
        return x
