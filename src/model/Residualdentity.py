from torch import nn


class ResidualIdentity(nn.Module):
    def __init__(self, channels):
        super(ResidualIdentity, self).__init__()

        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=True)

        self.conv_skip = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.bn_skip = nn.BatchNorm2d(channels)

    def forward(self, x):
        skip = self.conv_skip(x)
        skip = self.bn_skip(skip)

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)

        out += skip
        return out
