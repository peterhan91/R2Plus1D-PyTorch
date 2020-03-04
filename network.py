import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple

from module import SpatioTemporalConv # 2DConv --> BN+ReLU --> 1DConv 

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(ResBlock, self).__init__()
        self.downsample = downsample
        padding = kernel_size//2
        if self.downsample:
            self.downsampleconv =  nn.Conv3d(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        # STconv->batchnorm->ReLU->STconv->batchnorm->sum->ReLU
        res = self.relu1(self.bn1(self.conv1(x)))    
        res = self.bn2(self.conv2(res))
        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))
        return self.outrelu(x + res)

class SpatioTemporalResBottleBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBottleBlock, self).__init__()
        self.downsample = downsample
        padding = kernel_size//2
        self.conv0 = nn.Conv3d(in_channels, mid_channels, 1, stride=1)
        self.bn0 = nn.BatchNorm3d(mid_channels)
        self.relu0 = nn.ReLU()
        if self.downsample:
            self.downsampleconv = nn.Conv3d(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)
            self.conv1 = SpatioTemporalConv(mid_channels, mid_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = SpatioTemporalConv(mid_channels, mid_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(mid_channels, out_channels, 1, stride=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        res = self.relu0(self.bn0(self.conv0(x)))
        res = self.relu1(self.bn1(self.conv1(res)))    
        res = self.bn2(self.conv2(res))
        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))
        return self.outrelu(x + res)

class SpatioTemporalResBlock(nn.Module): # double STCONV Residual block
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in 
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)
        
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """
    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()
        
        # If downsample == True, the first conv of the layer has stride = 2 
        # to halve the residual output size, and the input x is passed 
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample
        
        # to allow for SAME padding
        padding = kernel_size//2

        if self.downsample:
            # downsample with stride =2 the input x
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, stride=2)
            self.downsamplebn = nn.BatchNorm3d(out_channels)

            # downsample with stride = 2when producing the residual
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding, stride=2)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        # STconv->batchnorm->ReLU->STconv->batchnorm->sum->ReLU
        res = self.relu1(self.bn1(self.conv1(x)))    
        res = self.bn2(self.conv2(res))
        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))
        return self.outrelu(x + res)


class SpatioTemporalResLayer(nn.Module): # a stack of RESBLOCKs
    r"""Forms a single layer of the ResNet network, with a number of repeating 
    blocks of same output size stacked on top of each other
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the layer.
            kernel_size (int or tuple): Size of the convolving kernels.
            layer_size (int): Number of blocks to be stacked to form the layer
            block_type (Module, optional): Type of block that is to be used to form the layer. Default: SpatioTemporalResBlock. 
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
        """
    def __init__(self, in_channels, out_channels, kernel_size, layer_size, 
                    block_type=SpatioTemporalResBlock, downsample=False): 
        super(SpatioTemporalResLayer, self).__init__()
        # implement the first block
        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)
        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]
    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x

class SpatioTemporalResBottleLayer(nn.Module): # a stack of RESBLOCKs
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size, layer_size, 
                    block_type=SpatioTemporalResBottleBlock, downsample=False): 
        super(SpatioTemporalResBottleLayer, self).__init__()
        self.block1 = block_type(in_channels, mid_channels, 
                                out_channels, kernel_size, downsample)
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            self.blocks += [block_type(out_channels, mid_channels, out_channels, kernel_size)]
    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)
        return x


class R2Plus1DNet(nn.Module):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in 
    each layer set by layer_sizes, and by performing a global average pool at the end producing a 
    512-dimensional vector for each element in the batch.
        
        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock. 
        """
    def __init__(self, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R2Plus1DNet, self).__init__()

        # first conv, with stride 1x2x2 and kernel size 3x7x7 ==> spatial output [112, 112]
        self.conv1 = SpatioTemporalConv(1, 32, [3, 7, 7], stride=[1, 2, 2], padding=[1, 3, 3])
        self.pool1 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2)) # ==> spatial output [56, 56]
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(32, 32, 3, layer_sizes[0], block_type=block_type)
        # each of the final three layers doubles num_channels, while performing downsampling 
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(32, 64, 3, layer_sizes[1], block_type=block_type, downsample=True)  # ==> spatial output [28, 28]
        self.conv4 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[2], block_type=block_type, downsample=True) # ==> spatial output [14, 14]
        self.conv5 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[3], block_type=block_type, downsample=True) # ==> spatial output [7, 7]
        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        return x.view(-1, 256)

class R2Plus1DBottleNet(nn.Module):
    def __init__(self, layer_sizes, CH=16, block_type=SpatioTemporalResBottleBlock):
        super(R2Plus1DBottleNet, self).__init__()
        self.CH = CH
        self.conv1 = SpatioTemporalConv(3, CH, [3, 7, 7],
                                        stride=[1, 2, 2], padding=[1, 3, 3])
        self.conv2 = SpatioTemporalResBottleLayer(CH, CH, CH*4, 3, layer_sizes[0], 
                                                    block_type=block_type, downsample=True)
        self.conv3 = SpatioTemporalResBottleLayer(CH*4, CH*2, CH*8, 3, layer_sizes[1], 
                                                    block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResBottleLayer(CH*8, CH*4, CH*16, 3, layer_sizes[2], 
                                                    block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResBottleLayer(CH*16, CH*8, CH*32, 3, layer_sizes[3], 
                                                    block_type=block_type, downsample=True) 
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        return x.view(-1, self.CH*32)

class R2Plus1DClassifier(nn.Module):
    r"""Forms a complete ResNet classifier producing vectors of size num_classes, by initializng 5 layers, 
    with the number of blocks in each layer set by layer_sizes, and by performing a global average pool
    at the end producing a 512-dimensional vector for each element in the batch, 
    and passing them through a Linear layer.
        
        Args:
            num_classes(int): Number of classes in the data
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock. 
        """
    def __init__(self, num_classes, layer_sizes, block_type=SpatioTemporalResBlock):
        super(R2Plus1DClassifier, self).__init__()

        self.res2plus1d = R2Plus1DNet(layer_sizes, block_type)
        # self.linear = nn.Linear(512, num_classes)
        self.linear = nn.Sequential(nn.Linear(512, num_classes), nn.Sigmoid())

    def forward(self, x):
        x = self.res2plus1d(x)
        x = self.linear(x) 

        return x   

class R2Plus1DBottleClassifier(nn.Module):
    def __init__(self, num_classes, layer_sizes, CH=16,
                    block_type=SpatioTemporalResBottleBlock):
        super(R2Plus1DBottleClassifier, self).__init__()
        self.res2plus1d = R2Plus1DBottleNet(layer_sizes, CH, block_type)
        self.linear = nn.Sequential(nn.Linear(CH*32, num_classes), 
                                    nn.Sigmoid())
    def forward(self, x):
        x = self.res2plus1d(x)
        x = self.linear(x) 
        return x  

class PARALLEClassifier(nn.Module):
    def __init__(self, num_classes, layer_sizes, 
                    block_type=ResBlock):
        super(PARALLEClassifier, self).__init__()
        self.res2plus1d_x = R2Plus1DNet(layer_sizes, block_type)
        self.res2plus1d_y = R2Plus1DNet(layer_sizes, block_type)
        self.res2plus1d_z = R2Plus1DNet(layer_sizes, block_type)
        self.linear = nn.Sequential(nn.Linear(256*3, num_classes), nn.Sigmoid())
    def forward(self, x, y, z):
        x1 = self.res2plus1d_x(x)
        y1 = self.res2plus1d_y(y)
        z1 = self.res2plus1d_z(z)
        features = torch.cat((x1, y1, z1), 1)
        out = self.linear(features)
        return out
        

