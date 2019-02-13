from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision


__all__ = ['VGG', 'vgg11','vgg13', 'vgg16', 'vgg19']


class VGG(nn.Module):
    __factory = {
        11: torchvision.models.vgg11_bn,
        13: torchvision.models.vgg13_bn,
        16: torchvision.models.vgg16_bn,
        19: torchvision.models.vgg19_bn,
    }

    def __init__(self, depth, pretrained=True, cut_at_last=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(VGG, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_last = cut_at_last

        # Construct base (pretrained) resnet
        if depth not in VGG.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = VGG.__factory[depth](pretrained=pretrained)
        print(self)

        if not self.cut_at_last:
            self.num_features = num_features
            self.norm = False#norm
            self.dropout = -1#dropout
            self.has_embedding = False#num_features > 0
            self.num_classes = num_classes

            out_planes = self.base.classifier[6].in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(
                    self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)

        if not self.pretrained:
            self.init_params()
        print(self)

    def forward(self, x):
        '''
        for name, module in self.base._modules.items():
            if name == 'classifier.6':
                break
            if name == 'classifier.0':
                x = x.view(x.size(0),-1)
            x = module(x)
        '''
        for name, module in self.base.named_modules():
            if not module._modules:
                if name == 'classifier.6':
                    break
                if name == 'classifier.0':
                    x = x.view(x.size(0),-1)
                x = module(x)

        if self.cut_at_last:
            return x

        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)
        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def vgg11(**kwargs):
    return VGG(11, **kwargs)


def vgg13(**kwargs):
    return VGG(13, **kwargs)


def vgg16(**kwargs):
    return VGG(16, **kwargs)


def vgg19(**kwargs):
    return VGG(19, **kwargs)
