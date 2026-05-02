import torch
import torch.nn as nn


class GaussianFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GaussianFusion, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out
