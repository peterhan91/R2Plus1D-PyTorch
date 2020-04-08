import math
from inflate_src import inflate
from inflate_src.self_local import NONLocalBlock3D
import torch
import torch.nn as nn
from torch.nn import ReplicationPad3d

center = False

class I3ResNet(torch.nn.Module):
    def __init__(self, resnet2d, class_nb=3, expansion=1):

        super(I3ResNet, self).__init__()
        self.expansion = expansion
        self.conv1 = inflate.inflate_conv(
            resnet2d.conv1, time_dim=3, time_padding=1, center=center)
        self.bn1 = inflate.inflate_batch_norm(resnet2d.bn1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = inflate.inflate_pool(
            resnet2d.maxpool, time_dim=3, time_padding=1, time_stride=1)

        self.layer1 = inflate_reslayer(resnet2d.layer1)
        self.layer2 = inflate_reslayer(resnet2d.layer2)
        self.layer3 = inflate_reslayer(resnet2d.layer3)
        self.layer4 = inflate_reslayer(resnet2d.layer4)
        
        self.non_local = NONLocalBlock3D(in_channels=128*self.expansion)
        # self.non_local_ = NONLocalBlock3D(in_channels=256*self.expansion)

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.linear = nn.Sequential(nn.Linear(512*self.expansion, class_nb), nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.non_local(x)
        x = self.layer3(x)
        # x = self.non_local_(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = self.linear(x.view(-1, 512*self.expansion))

        return x


def inflate_reslayer(reslayer2d, if_bottleneck=False):
    reslayers3d = []
    for layer2d in reslayer2d:
        if if_bottleneck:
            layer3d = Bottleneck3d(layer2d)
        else:
            layer3d = Basic3d(layer2d)
        reslayers3d.append(layer3d)
    return torch.nn.Sequential(*reslayers3d)

class Basic3d(torch.nn.Module):
    def __init__(self, basic2d):
        super(Basic3d, self).__init__()
        
        spatial_stride = basic2d.conv1.stride[0]

        self.conv1 = inflate.inflate_conv(
            basic2d.conv1,
            time_dim=3,
            time_padding=1,
            time_stride=1,
            center=center)
        self.bn1 = inflate.inflate_batch_norm(basic2d.bn1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = inflate.inflate_conv(
            basic2d.conv2,
            time_dim=3,
            time_padding=1,
            time_stride=1,
            center=center)
        self.bn2 = inflate.inflate_batch_norm(basic2d.bn2)

        if basic2d.downsample is not None:
            self.downsample = inflate_downsample(
                basic2d.downsample, time_stride=1)
        else:
            self.downsample = None

        self.stride = basic2d.stride

    def forward(self, x):
        
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class Bottleneck3d(torch.nn.Module):
    def __init__(self, bottleneck2d):
        super(Bottleneck3d, self).__init__()

        spatial_stride = bottleneck2d.conv2.stride[0]

        self.conv1 = inflate.inflate_conv(
            bottleneck2d.conv1, time_dim=1, center=center)
        self.bn1 = inflate.inflate_batch_norm(bottleneck2d.bn1)

        self.conv2 = inflate.inflate_conv(
            bottleneck2d.conv2,
            time_dim=3,
            time_padding=1,
            time_stride=1,
            center=center)
        self.bn2 = inflate.inflate_batch_norm(bottleneck2d.bn2)

        self.conv3 = inflate.inflate_conv(
            bottleneck2d.conv3, time_dim=1, center=center)
        self.bn3 = inflate.inflate_batch_norm(bottleneck2d.bn3)

        self.relu = torch.nn.ReLU(inplace=True)

        if bottleneck2d.downsample is not None:
            self.downsample = inflate_downsample(
                bottleneck2d.downsample, time_stride=1)
        else:
            self.downsample = None

        self.stride = bottleneck2d.stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


def inflate_downsample(downsample2d, time_stride=1):
    downsample3d = torch.nn.Sequential(
        inflate.inflate_conv(
            downsample2d[0], time_dim=1, time_stride=time_stride, center=center),
        inflate.inflate_batch_norm(downsample2d[1]))
    return downsample3d
