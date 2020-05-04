from collections import OrderedDict

import torch.nn as nn
import torch

__all__ = ['resnet101']


class Bottleneck(nn.Module):
    """
    Base class for Bottlenecks.
    'forward()' method has been implemented. 
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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
        out = out + residual
        out = self.relu(out)

        return out



class ResNet(nn.Module):
    def __init__(self, block, layers, groups, reduction, dropout_p=0.2, inplanes=128, input_3x3=True, downsample_kernel_size=3, downsample_padding=1, total_len=512, M=4, attention=True):
        """
        groups (int): Number of groups for the 3x3 convolution in each bottleneck block.
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of a single 7x7 convolution in layer0.
        """
        super(ResNet, self).__init__()
        self.inplanes = inplanes
        self.attention = attention
        self.M = M

        if input_3x3:
            layer0_modules = [
                    ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False)),
                    ('bn1', nn.BatchNorm2d(64)),
                    ('relu1', nn.ReLU(inplace=True)),
                    ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)),
                    ('bn2', nn.BatchNorm2d(64)),
                    ('relu2', nn.ReLU(inplace=True)),
                    ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1, bias=False)),
                    ('bn3', nn.BatchNorm2d(inplanes)),
                    ('relu3', nn.ReLU(inplace=True)),
                    ]
        else:
            layer0_modules = [
                    ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)),
                    ('bn1', nn.BatchNorm2d(inplanes)),
                    ('relu1', nn.ReLU(inplace=True)),                            
                    ]
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)))
        self.layers0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layers1 = self._make_layer(
                block,
                planes=64,
                blocks=layers[0],
                groups=groups,
                reduction=reduction,
                downsample_kernel_size=1,
                downsample_padding=0
                )
        self.layers2 = self._make_layer(
                block,
                planes=128,
                blocks=layers[1],
                stride=2,
                groups=groups,
                reduction=reduction,
                downsample_kernel_size=downsample_kernel_size,
                downsample_padding=downsample_padding 
                )
        self.layers3 = self._make_layer(
                block,
                planes=256,
                blocks=layers[2],
                stride=2,
                groups=groups,
                reduction=reduction,
                downsample_kernel_size=downsample_kernel_size,
                downsample_padding=downsample_padding 
                )
        if attention == True:
            self.att_layer = self._make_layer(
                    block,
                    planes=512,
                    blocks=layers[3],
                    stride=1,  # to keep the same size
                    groups=groups,
                    reduction=reduction,
                    downsample_kernel_size=downsample_kernel_size,
                    downsample_padding=downsample_padding,
                    )
            self.att_branches = []
            for i in range(self.M):
                self.att_branches.append(nn.Sequential(
                        nn.Conv2d(self.inplanes, self.att_inplanes, kernel_size=1, padding=0, bias=False),
                        nn.BatchNorm2d(self.att_inplanes),
                        nn.Sigmoid()
                        ))
            self.att_branches = nn.ModuleList(self.att_branches)
            self.inplanes = self.att_inplanes

        self.layers4 = self._make_layer(
                block,
                planes=512,
                blocks=layers[3],
                stride=2,
                groups=groups,
                reduction=reduction,
                downsample_kernel_size=downsample_kernel_size,
                downsample_padding=downsample_padding 
                )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512*block.expansion, int(total_len/self.M))
       
    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1, downsample_kernel_size=1, downsample_padding=0):
        self.att_inplanes = self.inplanes 
        downsample = None
        if stride!=1 or self.inplanes!=planes*block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=downsample_kernel_size, stride=stride, padding=downsample_padding, bias=False),
                    nn.BatchNorm2d(planes*block.expansion),
                    )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def feature(self, x):
        x = self.layers0(x)
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        att_outputs = []
        for i in range(self.M):
            att_out = self.att_layer(x)
            att_mask = self.att_branches[i](att_out)
            att_output = torch.mul(x, att_mask)
            att_output = self.layers4(att_output)
            att_outputs.append(att_output)
        return att_outputs
    
    def forward(self, x):
        x = self.feature(x)
        for i in range(self.M):
            x[i] = self.avg_pool(x[i])
            if self.dropout is not None:
                x[i] = self.dropout(x[i])
            x[i] = x[i].view(x[i].size(0), -1)
            x[i] = self.last_linear(x[i])
            norm = x[i].norm(dim=1, p=2, keepdim=True)
            x[i] = x[i].div(norm.expand_as(x[i]))
        
        return x #list


#========================================== models =========================================

def resnet101(total_len=512, M=4, attention=True):
    model = ResNet(Bottleneck, [3, 4, 6, 3], groups=1, reduction=16, dropout_p=None, inplanes=64, input_3x3=False, downsample_kernel_size=1, downsample_padding=0, total_len=total_len, attention=attention)
    return model


