# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T
import random
import numpy as np
import PIL
from .transforms import RandomErasing

class AddGaussianNoise(object):
    def __call__(self, img):
        std = random.uniform(0, 1.0)
        if std > 0.5:
            return img

        # Convert to ndarray
        img = np.asarray(img).copy()
        noise = np.random.normal(size=img.shape, scale=std).astype(np.uint8)
        img += noise
        img = np.clip(img, 0, 255)

        # Convert back to PIL image
        img = PIL.Image.fromarray(img)
        return img

def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        print('++++  hard    train')
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_DOWN),
            T.Resize(cfg.INPUT.SIZE_UP),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(padding=cfg.INPUT.PADDING),
            T.RandomRotation(cfg.INPUT.DEGREE),
            T.ColorJitter(0.6,0.9,0.7),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            #AddGaussianNoise(),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        print('++++  init    test')
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform

def build_transforms2(cfg, is_train=True):
    #print('++++  easy')
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        print('++++  easy    train')
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            #T.Pad(cfg.INPUT.PADDING),
            T.ColorJitter(0.4,0.6,0.7),
            T.RandomRotation(cfg.INPUT.DEGREE),
            #T.ColorJitter(0.4,0.6,0.7),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        print('++++  easy    test')
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform

def build_transforms3(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        print('++++  init    train')
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        print('++++  init    test')
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
