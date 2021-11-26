"""
Creates an Xception Model as defined in:

Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf

This weights ported from the Keras implementation. Achieves the following performance on the validation set:

Loss:0.9173 Prec@1:78.892 Prec@5:94.292

REMEMBER to set your image size to 3x299x299 for both test and validation

normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])

The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch

__all__ = ['xception']

model_urls = {
    'xception': 'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
}


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True, BatchNorm=None):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = BatchNorm(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(BatchNorm(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(BatchNorm(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(BatchNorm(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, BatchNorm,
                 pretrained=False):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        # self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = BatchNorm(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = BatchNorm(64)
        # do relu here

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True, BatchNorm=BatchNorm)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True,BatchNorm=BatchNorm )
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True,BatchNorm=BatchNorm)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True,BatchNorm=BatchNorm)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True,BatchNorm=BatchNorm)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True,BatchNorm=BatchNorm)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True,BatchNorm=BatchNorm)

        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True,BatchNorm=BatchNorm)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True,BatchNorm=BatchNorm)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True,BatchNorm=BatchNorm)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True,BatchNorm=BatchNorm)

        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False,BatchNorm=BatchNorm)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = BatchNorm(2048)

        self.fc = nn.Linear(2048, 1000)

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------
        if pretrained:
            self._init_weights()

    def forward(self, x):
        output_encoder = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        low_level_feat = x
        output1 =x
        output_encoder.append(output1)
        x = self.block2(x)
        output2=x
        output_encoder.append(output2)
        x = self.block3(x)
        output3=x
        output_encoder.append(output3)

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        output13 =x
        output_encoder.append(output13)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        output14= x
        output_encoder.append(output14)
        x = self.relu(x)
        #
        # x = F.adaptive_avg_pool2d(x, (1, 1))
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x,low_level_feat, output_encoder

    def _init_weights(self):
        pretrain_dict = model_zoo.load_url(model_urls['xception'])
        # model_dict = {}
        state_dict = self.state_dict()
        state_dict.update(pretrain_dict)
        self.load_state_dict(state_dict)

def xception(pretrained=False, **kwargs):
    """
    Construct Xception.
    """

    model = Xception(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['xception']))
    return model