import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from .res_decoder import _ResDecBlockTrans

MOMENTUM = 0.01
NEGATIVE_SLOPE = 0.01


class _ConvTrans2dBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 use_bn=True, use_relu=True):
        super().__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu

        self.conv_trans_2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, momentum=MOMENTUM) if use_bn else None
        self.relu = nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE, inplace=True) if use_relu else None

    def forward(self, x):
        x = self.conv_trans_2d(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_relu:
            x = self.relu(x)
        return x


class _FCBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels, momentum=MOMENTUM),
            nn.LeakyReLU(negative_slope=NEGATIVE_SLOPE, inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Decoder(nn.Module):
    """
    -- 通过蒸馏损失控制Decoder遗忘
    """

    def __init__(self, in_dim=128, z_dim=64, out_dim=3):
        """
        ConvTranspose2d: H_out = (H_in - 1) * stride - 2 * padding + ker_sz + output_padding
        :param in_dim: decoder input dimension
        :param z_dim: latent dimension
        :param out_dim: decoder output dimension (image dimension)
        :param out_size: decoder output height (width)
        """
        super().__init__()

        self.res_blocks = nn.Sequential(
            _ResDecBlockTrans(in_dim, in_dim),
            _ResDecBlockTrans(in_dim, in_dim),
        )

        self.fc_blocks = nn.Sequential(
            _FCBlock(in_dim, 512),
            _FCBlock(512, 128 * 3 * 3),
        )
        self.conv_trans_blocks = nn.Sequential(
            # _ResDecBlockTrans(128, 128),
            _ConvTrans2dBlock(128, 64, kernel_size=3, stride=1, padding=0),
            # _ResDecBlockTrans(64, 64),
            _ConvTrans2dBlock(64, 32, kernel_size=4, stride=2, padding=1),
            # _ResDecBlockTrans(32, 32),
            _ConvTrans2dBlock(32, 16, kernel_size=4, stride=2, padding=1),
            # _ResDecBlockTrans(16, 16),
            _ConvTrans2dBlock(16, 8, kernel_size=4, stride=2, padding=1),
            # _ResDecBlockTrans(8, 8),
            _ConvTrans2dBlock(8, out_dim, kernel_size=4, stride=2, padding=1,
                              use_bn=True, use_relu=False),
            nn.Sigmoid(),
        )

    def forward(self, x, out_size=84):
        """
        :param x: [bs, in_dim, 3, 3]
        :param out_size: default 84
        :return: [bs, out_dim, out_size, out_size]
        """
        x = self.res_blocks(x)
        # print(x.shape)
        # x = x.flatten(start_dim=1)
        # x = self.fc_blocks(x)
        # x = x.view(-1, 128, 3, 3)
        x = self.conv_trans_blocks(x)
        # print(x.shape)
        x = F.interpolate(x, size=(out_size, out_size),
                          mode='bilinear', align_corners=True)
        return x


def debug():
    model = Decoder(in_dim=128, z_dim=64, out_dim=3)
    # print(model2)

    data = torch.randn(2, 128, 3, 3)
    output = model(data)
    print(output.shape)  # [2, 3, 84, 84]


if __name__ == '__main__':
    debug()
