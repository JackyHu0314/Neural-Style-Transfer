import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride),
        )

    def forward(self, x):
        return self.block(x)


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="nearest"),
            ConvLayer(in_channels, out_channels, kernel_size),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvLayer(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            ConvLayer(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)


class TransformerNet(nn.Module):
    """Feed-forward image transformation network.

    Input and output tensors use RGB values in [0, 1].
    """

    def __init__(self, residual_blocks=5):
        super().__init__()
        layers = [
            ConvLayer(3, 32, kernel_size=9),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            ConvLayer(32, 64, kernel_size=3, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            ConvLayer(64, 128, kernel_size=3, stride=2),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
        ]

        for _ in range(residual_blocks):
            layers.append(ResidualBlock(128))

        layers.extend(
            [
                UpsampleConvLayer(128, 64, kernel_size=3),
                nn.InstanceNorm2d(64, affine=True),
                nn.ReLU(inplace=True),
                UpsampleConvLayer(64, 32, kernel_size=3),
                nn.InstanceNorm2d(32, affine=True),
                nn.ReLU(inplace=True),
                ConvLayer(32, 3, kernel_size=9),
                nn.Sigmoid(),
            ]
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def total_variation_loss(image):
    """Small smoothness penalty that reduces noisy artifacts."""
    horizontal = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    vertical = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    return horizontal + vertical

