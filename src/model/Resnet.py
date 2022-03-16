from torch import nn

from src.model.Conv import Conv
from src.model.ResidualPaddedDownsample import ResidualPaddedDownsample
from src.model.Residualdentity import ResidualIdentity


class Resnet(nn.Module):
    def __init__(self, in_channels, out_size, data_shape):
        super(Resnet, self).__init__()
        self.in_channels = in_channels
        self.out_size = out_size

        self.pre_conv = Conv(in_channels, 16, 3, 1, bn=True, relu=True)

        self.residual_downsample1 = ResidualPaddedDownsample(16, 32)
        self.residual_identity1 = ResidualIdentity(32)
        self.residual_downsample2 = ResidualPaddedDownsample(32, 64)
        self.residual_identity2 = ResidualIdentity(64)
        self.residual_downsample3 = ResidualPaddedDownsample(64, 128)
        self.residual_identity3 = ResidualIdentity(128)

        linear_input_size = 128 * (data_shape[0] // 8) * (data_shape[1] // 8)
        self.linear_out = nn.Linear(linear_input_size, out_size)
        self.bn_out = nn.BatchNorm1d(out_size)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.residual_downsample1(x)
        x = self.residual_identity1(x)
        x = self.residual_downsample2(x)
        x = self.residual_identity2(x)
        x = self.residual_downsample3(x)
        x = self.residual_identity3(x)

        # x = x.flatten(1)
        x = x.view(x.size(0), -1)
        x = self.linear_out(x)
        x = self.bn_out(x)
        return x
