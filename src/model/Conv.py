from torch import nn


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bn=False, relu=False):
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = relu
        self.bn = bn
        if relu:
            self.reluF = nn.ReLU()
        if bn:
            self.bnF = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bnF(x)
        if self.relu:
            x = self.reluF(x)
        return x
