import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + x)

class ChessPolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(18, 256, 3, padding=1)
        self.bn = nn.BatchNorm2d(256)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(10)]
        )

        self.policy = nn.Conv2d(256, 73, 1)
        self.fc = nn.Linear(73 * 8 * 8, 4672)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        x = self.res_blocks(x)
        x = self.policy(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
