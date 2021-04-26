import torch
import torch.nn as nn
import torch.nn.functional as F

MOMENTUM = 0.01
NEGATIVE_SLOPE = 0.01


class _ResDecBlockTrans(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        if stride == 1:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.shortcut = None
        else:
            self.conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=stride, padding=1,
                                            output_padding=1, bias=False)
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(inplanes, planes, kernel_size=1,
                                   stride=stride, output_padding=1, bias=False),
                nn.BatchNorm2d(planes),
            )
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        # print('conv1', out.shape)
        out = self.bn2(self.conv2(out))
        # print('conv2', out.shape)

        if self.shortcut is not None:
            identity = self.shortcut(x)
        out += identity
        out = self.relu(out)

        return out


class _InterpConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor,
                 kernel_size=3, padding=1, mode='bilinear'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corner = True if mode == 'bilinear' else None
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=1, padding=padding, bias=False)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode,
                          align_corners=self.align_corner)
        x = self.conv(x)
        return x


class _ResDecBlockInterp(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        if stride == 1:
            self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.shortcut = None
        else:
            self.conv2 = _InterpConv2d(inplanes, planes, scale_factor=stride)
            self.shortcut = nn.Sequential(
                _InterpConv2d(inplanes, planes, kernel_size=1, scale_factor=stride, padding=0),
                nn.BatchNorm2d(planes)
            )
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))

        if self.shortcut is not None:
            identity = self.shortcut(x)
        out += identity
        out = self.relu(out)

        return out


class _ResDecoder(nn.Module):

    def __init__(self, block, n_layers, in_dim=128, out_dim=3):
        """
        in: [bs, 512, 3, 3]; out: [bs, 3, 84, 84]
        :param block:
        :param n_layers:
        :param in_dim:
        :param out_dim:
        """
        super().__init__()
        self.inplanes = in_dim
        n_planes = [in_dim, in_dim // 2, in_dim // 4, in_dim // 8]

        self.layer1 = self._make_layer(block, n_layers[0], n_planes[0], stride=2)
        self.layer2 = self._make_layer(block, n_layers[1], n_planes[1], stride=2)
        self.layer3 = self._make_layer(block, n_layers[2], n_planes[2], stride=2)
        self.layer4 = self._make_layer(block, n_layers[3], n_planes[3], stride=2)

        self.layer_top = nn.Sequential(
            nn.ConvTranspose2d(n_planes[3], out_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Tanh(),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, n_block, planes, stride=1):
        # upsample = None
        # if stride != 1 or self.inplanes != planes:
        #     upsample = nn.Sequential(
        #         nn.ConvTranspose2d(self.inplanes, planes, kernel_size=1,
        #                            stride=stride, output_padding=1, bias=False),
        #         nn.BatchNorm2d(planes),
        #     )
        layers = [block(self.inplanes, planes, stride=stride)]
        self.inplanes = planes
        for i in range(1, n_block):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, scale_factor=None, out_size=32):
        x = self.layer1(x)
        # print('x1', x.shape)
        x = self.layer2(x)
        # print('x2', x.shape)
        x = self.layer3(x)
        # print('x3', x.shape)
        x = self.layer4(x)
        # print('x4', x.shape)
        x = self.layer_top(x)
        # print('x5', x.shape)
        x = F.interpolate(x, size=(out_size, out_size),
                          mode='bilinear', align_corners=True)
        return x


class _ResDecoderFM3(nn.Module):

    def __init__(self, block, n_layers, in_dim=512, out_dim=256):
        """
        in: [bs, 512, 1, 1]; out: [bs, 256, 6, 6]
        """
        super().__init__()
        self.inplanes = in_dim
        self.layer1 = self._make_layer(block, n_layers[0], 256, stride=2)
        self.layer_top = nn.Sequential(
            nn.Conv2d(256, out_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(inplace=True),
            # nn.Tanh(),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, n_block, planes, stride=1):
        layers = [block(self.inplanes, planes, stride=stride)]
        self.inplanes = planes
        for i in range(1, n_block):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, scale_factor=3, out_size=6):
        if scale_factor > 1:
            x = F.interpolate(x, scale_factor=scale_factor)
        x = self.layer1(x)
        x = self.layer_top(x)
        # print(x.shape)
        x = F.interpolate(x, size=(out_size, out_size),
                          mode='bilinear', align_corners=True)
        return x


def res_decoder18(in_dim, out_dim, fm_level=-1):
    if fm_level == -1:
        return _ResDecoder(_ResDecBlockInterp, [2, 2, 2, 2], in_dim=in_dim, out_dim=out_dim)
    elif fm_level == 3:
        return _ResDecoderFM3(_ResDecBlockInterp, [2], in_dim=in_dim, out_dim=out_dim)
    else:
        raise NotImplementedError('fm_level {} not implemented.'.format(fm_level))


def debug():
    model = _ResDecoderFM3(_ResDecBlockInterp, [2], in_dim=512, out_dim=256).cuda()
    # model = _ResDecoder(_ResDecBlockInterp, [2, 2, 2, 2], in_dim=512, out_dim=3).cuda()
    print(model)

    data = torch.randn(2, 512, 1, 1).cuda()

    with torch.no_grad():
        output = model(data, scale_factor=3)
        print(output.shape)  # [2, 3, 84, 84]


if __name__ == '__main__':
    debug()
