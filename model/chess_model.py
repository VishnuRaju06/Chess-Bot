import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class ChessModel(nn.Module):
    def __init__(
        self,
        in_channels=18,
        channels=256,
        num_blocks=10,
        num_moves=4672
    ):
        super().__init__()

        # Initial convolution
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)

        # Residual tower
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # Policy head
        self.policy = nn.Conv2d(channels, 73, kernel_size=1)
        self.fc = nn.Linear(73 * 8 * 8, num_moves)

    def forward(self, x):
        # x: (B, 18, 8, 8)
        x = F.relu(self.bn(self.conv(x)))

        for block in self.res_blocks:
            x = block(x)

        x = self.policy(x)               # (B, 73, 8, 8)
        x = x.view(x.size(0), -1)        # (B, 4672)
        x = self.fc(x)                   # (B, 4672)

        return x
