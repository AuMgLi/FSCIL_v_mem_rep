import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class ECALayer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)  # [b, c, 1, 1]
        # print('y:', y.shape)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class ECABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = ECALayer(planes, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class BasicBlock4ResNet12(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False,
                 block_size=1, use_se=False):
        super(BasicBlock4ResNet12, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        # self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        # self.DropBlock = DropBlock(block_size=self.block_size) if drop_rate > 0 else None
        self.use_se = use_se
        self.se = SELayer(planes, 4) if use_se else None

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.use_se:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        # print('residual_:', residual.shape, 'out:', out.shape)
        out += residual
        out = self.relu(out)
        # out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block is True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * self.num_batches_tracked, 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, activation=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = nn.LeakyReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.activation = activation
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, n_blocks, n_classes=None,
                 use_avgpool=True, out_channels=None, is_snail=False):
        super().__init__()

        self.inplanes = 32 if is_snail else 64
        self.use_avgpool = use_avgpool

        self.downsample = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            nn.BatchNorm2d(self.inplanes),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Todo use in face only
        )
        if is_snail:
            n_planes = [32, 64, 128, 256]
        else:
            n_planes = [64, 128, 256, 512]

        # w/o. drop_block?
        self.layer1 = self._make_layer(block, n_blocks[0], n_planes[0], stride=1)
        self.layer2 = self._make_layer(block, n_blocks[1], n_planes[1], stride=2)
        self.layer3 = self._make_layer(block, n_blocks[2], n_planes[2], stride=2,)  # activation=nn.Tanh()
        self.layer4 = self._make_layer(block, n_blocks[3], n_planes[3], stride=2)

        self.n_classes = n_classes
        if out_channels is None:
            self.out_channels = n_planes[3] * block.expansion
        else:
            self.out_channels = out_channels * block.expansion
        if use_avgpool:
            self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        if n_planes[3] != self.out_channels:  # transform channel dim to memory key dim
            self.channel_transform = nn.Sequential(
                nn.Conv2d(n_planes[3], out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        if n_classes is not None:  # use classifier
            self.classifier = nn.Linear(self.out_channels, self.n_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, n_block, planes, stride=1, activation=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, activation=activation)]
        self.inplanes = planes * block.expansion
        for i in range(1, n_block):
            layer = block(self.inplanes, planes, activation=activation)
            layers.append(layer)

        return nn.Sequential(*layers)

    def forward(self, x, return_fm=False, feed_fm=False):
        if feed_fm:
            x = self.layer4(x)
            if x.size(1) != self.out_channels:
                x = self.channel_transform(x)  # [, oc, 3, 3]
            if self.use_avgpool:
                x = self.avg_pool(x)  # [, oc, 1, 1]
            return x
        else:
            x = self.downsample(x)
            # print('2', x.shape)

            x = self.layer1(x)  # [, 32|64, 21, 21]
            fm1 = x
            x = self.layer2(x)  # [, 64|128, 11, 11]
            fm2 = x
            x = self.layer3(x)  # [, 128|256, 6, 6]
            fm3 = x
            x = self.layer4(x)  # [, 256|512, 3, 3]
            # x = F.normalize(x, p=2, dim=1)
            fm4 = x
            if x.size(1) != self.out_channels:
                x = self.channel_transform(x)  # [, oc, 3, 3]
            if self.use_avgpool:
                x = self.avg_pool(x)  # [, oc, 1, 1]
                fm4 = F.normalize(x, p=2, dim=1)
            if self.n_classes is not None:
                x = torch.flatten(x, 1)
                x = self.classifier(x)
            if return_fm:
                return x, (fm1, fm2, fm3, fm4)
            else:
                return x


def resnet12(**kwargs):
    """Constructs a ResNet-12 encoder.
    """
    model = ResNet(BasicBlock4ResNet12, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """
    ResNet args: block, n_blocks, drop_rate=0.1, dropblock_size=5, n_classes=None,
                 use_avgpool=True, use_se=False, out_channels=None
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def debug():
    model = resnet18(n_classes=None, use_avgpool=True, out_channels=None, is_snail=False)
    data = torch.randn(2, 3, 128, 128)
    model = model.cuda()
    data = data.cuda()
    x, fms = model(data, return_fm=True)
    print('x:', x.shape,)
    for fm in fms:
        print(fm.shape)


if __name__ == '__main__':
    debug()
