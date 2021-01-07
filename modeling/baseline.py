# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
import numpy as np
#from .AdMSLoss import AdMSoftmaxLoss


class Arcface(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=64.0, m=0.2, easy_margin=False):
        super(Arcface, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)
        return output


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def feature_erasing(feature, ratio=0.2, is_training=True):
    if not is_training or ratio == 0:
        return feature
    n, c = feature.shape
    era_num = int(ratio * c)

    mask = torch.ones_like(feature)

    rand_index = np.random.randint(0, high=c, size=era_num)
    mask[:,rand_index] = 0
    feature = feature * mask
    return feature


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, num_classes2, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Baseline, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            print('\n\n************  resnet50 \n\n')
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            print('\n\n************  resnet101 \n\n')
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])

        elif model_name == 'se_resnet50':
            print('\n\n************  se_resnet50 \n\n')
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet101':
            print('\n\n************  se_resnet101 \n\n')
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=1,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext50':
            print('\n\n************  se_resnext50 \n\n')
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnext101':
            print('\n\n************  se_resnext101 \n\n')
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3],
                              groups=32,
                              reduction=16,
                              dropout_p=None,
                              inplanes=64,
                              input_3x3=False,
                              downsample_kernel_size=1,
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck,
                              layers=[3, 8, 36, 3],
                              groups=64,
                              reduction=16,
                              dropout_p=0.2,
                              last_stride=last_stride)

        if pretrain_choice == 'imagenet':
            print('Loading pretrained ImageNet model......')
            self.base.load_param(model_path)
            #print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.share = False
        self.arcface = False
        self.feat_erase = False
        self.feat_erase_pos = 'before'
        self.feat_erase_ratio = 0.2
        print('self.share:', self.share)
        print('self.arcface:', self.arcface)
        print('self.feat_erase:', self.feat_erase)
        print('self.feat_erase_pos:', self.feat_erase_pos)
        print('self.feat_erase_ratio:', self.feat_erase_ratio)

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.bottleneck.apply(weights_init_kaiming)

            if self.share:
                print('-----------')
                self.bottleneck2 = nn.BatchNorm1d(self.in_planes)
                self.bottleneck2.bias.requires_grad_(False)  # no shift

            if self.arcface:
              self.classifier = Arcface(self.in_planes, self.num_classes)
            else:
              self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
              self.classifier.apply(weights_init_classifier)
              #self.classifier= AdMSoftmaxLoss(self.in_planes, num_classes, s=16.0, m=1)

            self.classifier2 = nn.Linear(self.in_planes, num_classes2, bias=False)
            self.classifier2.apply(weights_init_classifier)

    def forward(self, x, y=None, flag=False):

        global_feat0 = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat0.view(global_feat0.shape[0], -1)  # flatten to (bs, 2048)
        if self.share:
            global_feat2 = global_feat0.view(global_feat0.shape[0], -1)

        if self.neck == 'no':
            feat = global_feat
            if self.share:
                feat2 = global_feat2
        elif self.neck == 'bnneck':
            if self.feat_erase and self.feat_erase_pos == 'before':
                global_feat = feature_erasing(global_feat, ratio=0.5, is_training=self.training)
            feat = self.bottleneck(global_feat)  # normalize for angular softmax
            if self.feat_erase and self.feat_erase_pos == 'after':
                feat = feature_erasing(feat, ratio=self.feat_erase_ratio, is_training=self.training)
            if self.share:
                feat2 = self.bottleneck2(global_feat2)

        if self.training:
            if self.arcface:
              cls_score = self.classifier(feat, y)
            else:
              if flag:
                cls_score = self.classifier(feat, y)
              else:
                cls_score = self.classifier(feat)
            if self.share:
                cls_score2 = self.classifier2(feat2)
            else:
                cls_score2 = self.classifier2(feat)
            #print(' ok-1')
            return cls_score, cls_score2, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        print('[baseline]  Imagenet pretrained')
        param_dict = torch.load(trained_path)
        #print('loading from ', trained_path)
        #print('++++++++++++++++\n\n')
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        #print('[baseline]  Imagenet pretrained')

        '''
        ckpt = torch.load(trained_path, map_location=lambda storage, loc: storage)
        print('---------')
        print(ckpt)
        if 'state_dicts' in ckpt:
            state_dict = ckpt['state_dicts'][0]
        else:
            state_dict = ckpt['state_dict']
        print('*******')
        print(state_dict)
        return 0
        for i in param_dict:
            if 'classifier' in i:
                continue
            try:
                self.state_dict()[i].copy_(param_dict[i])
            except:
                print()
        '''
