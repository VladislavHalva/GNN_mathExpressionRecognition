import torch.nn.functional
from torch import nn
import torch.nn.functional as F

class ResidualPaddedDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualPaddedDownsample, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 2, 1, bias=True)
        self.skip_pool = nn.MaxPool2d(2)

    def forward(self, x):
        channels_to_pad = self.out_channels - self.in_channels

        skip = self.skip_pool(x)
        skip = F.pad(skip, (0, 0, 0, 0, channels_to_pad // 2, channels_to_pad // 2), 'constant', 0)

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)

        out += skip
        return out



