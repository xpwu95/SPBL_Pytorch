from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision


__all__ = ['DenseNet', 'densenet121', 'densenet161',
           'densenet169', 'densenet201']


class DenseNet(nn.Module):
    __factory = {
        121: torchvision.models.densenet121,
        169: torchvision.models.densenet169,
        161: torchvision.models.densenet161,
        201: torchvision.models.densenet201,
    }

    def __init__(self, depth, pretrained=True, cut_at_last=False,
                 num_features=0, norm=False, dropout=0, num_classes=0):
        super(DenseNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_last = cut_at_last

        # Construct base (pretrained) resnet
        if depth not in DenseNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = DenseNet.__factory[depth](pretrained=pretrained)

        out_num_features = self.base.features.norm5.num_features
        self.num_classes = num_classes

        if not self.cut_at_last:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_num_features, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_num_features
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
        if self.num_classes > 0:
            self.classifier = nn.Linear(self.num_features, self.num_classes)
            init.normal(self.classifier.weight, std=0.001)
            init.constant(self.classifier.bias, 0)

        if not self.pretrained:
            self.init_params()

    def forward(self, x):
        x = self.base.features(x)
        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)

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


def densenet121(**kwargs):
    return DenseNet(121, **kwargs)


def densenet161(**kwargs):
    return DenseNet(161, **kwargs)


def densenet169(**kwargs):
    return DenseNet(169, **kwargs)


def densenet201(**kwargs):
    return DenseNet(201, **kwargs)
