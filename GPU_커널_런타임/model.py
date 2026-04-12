from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import plugin_stub  # noqa: F401


class AlexNetImageToImage(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 3, use_plugin: bool = True):
        super().__init__()
        self.use_plugin = use_plugin

        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.dec1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.out_proj = nn.Conv2d(128, out_ch, kernel_size=3, padding=1)

    def _conv1_block(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_plugin:
            return torch.ops.myimg.alex_conv1_plugin_stub(
                x,
                self.conv1.weight,
                self.conv1.bias,
                4,
                4,
                2,
                2,
            )

        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_hw = x.shape[-2:]

        x = self._conv1_block(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=3, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = F.relu(self.dec1(x))
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = F.relu(self.dec2(x))
        x = F.interpolate(x, size=target_hw, mode="nearest")
        return self.out_proj(x)
