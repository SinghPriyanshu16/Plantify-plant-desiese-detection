import torch
import torch.nn as nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import misc as misc_nn_ops
import torchvision

from .attention import SelfAttention

class DilatedBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=32,
                 base_width=4, dilation=1, norm_layer=None):
        super(DilatedBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, groups=groups, bias=False)
        self.bn2 = norm_layer(width)
        
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class CustomResNeXt(nn.Module):
    def __init__(self, block, layers, groups=32, width_per_group=4, replace_stride_with_dilation=None):
        super(CustomResNeXt, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        
        # Add attention before layer4
        self.attention = SelfAttention(256 * block.expansion)
        
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c4 = self.attention(c4)
        c5 = self.layer4(c4)

        return {"0": c2, "1": c3, "2": c4, "3": c5}

def get_daa_mrcnn(num_classes):
    """
    Constructs the DAA-MRCNN (Dilated, Adaptive, and Attention-based Mask RCNN)
    using a custom ResNeXt-101 backbone with dilated convolutions and attention.
    """
    # ResNeXt-101 layers: [3, 4, 23, 3]
    backbone = CustomResNeXt(DilatedBottleneck, [3, 4, 23, 3], groups=32, width_per_group=8,
                             replace_stride_with_dilation=[False, True, True])
    
    # Feature Pyramid Network requires defined returned channels
    out_channels = 256
    in_channels_list = [256, 512, 1024, 2048]
    
    from torchvision.ops import FeaturePyramidNetwork
    fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels)
    
    class BackboneWithFPN(nn.Module):
        def __init__(self, backbone, fpn):
            super().__init__()
            self.body = backbone
            self.fpn = fpn
            self.out_channels = out_channels

        def forward(self, x):
            x = self.body(x)
            x = self.fpn(x)
            return x

    backbone_with_fpn = BackboneWithFPN(backbone, fpn)
    
    # Instantiate Mask RCNN
    model = MaskRCNN(backbone_with_fpn, num_classes=num_classes)
    return model
