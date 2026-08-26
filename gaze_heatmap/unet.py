# Adapted from peg-in-hole-visual-servoing-model/unet.py (ResNet18-UNet).
from __future__ import annotations

import torch
from torch import nn
import torchvision


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, n_class: int, feat_preultimate: int = 64):
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights

            weights = ResNet18_Weights.DEFAULT
        except ImportError:
            weights = None
        self.base_model = torchvision.models.resnet18(weights=weights)
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        self.conv_original_size2 = convrelu(128, feat_preultimate, 3, 1)
        self.conv_last = nn.Conv2d(feat_preultimate, n_class, 1)
        for p in self.base_model.conv1.parameters():
            p.requires_grad = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        x = self.layer4_1x1(layer4)
        x = self.upsample(x)
        x = torch.cat([x, self.layer3_1x1(layer3)], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        x = torch.cat([x, self.layer2_1x1(layer2)], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        x = torch.cat([x, self.layer1_1x1(layer1)], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        x = torch.cat([x, self.layer0_1x1(layer0)], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = self.conv_original_size2(x)
        return self.conv_last(x)
