import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, t1, t2, t3, t4):
        for t in self.transforms:
            image, t1, t2, t3, t4 = t(image, t1, t2, t3, t4)
        return image, t1, t2, t3, t4


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, t1, t2, t3, t4):
        size = random.randint(self.min_size, self.max_size)
        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image = F.resize(image, size)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        t1 = F.resize(t1, size, interpolation=T.InterpolationMode.NEAREST)
        t2 = F.resize(t2, size, interpolation=T.InterpolationMode.NEAREST)
        t3 = F.resize(t3, size, interpolation=T.InterpolationMode.NEAREST)
        t4 = F.resize(t4, size, interpolation=T.InterpolationMode.NEAREST)
        return image, t1, t2, t3, t4


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, t1, t2, t3, t4):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            t1 = F.hflip(t1)
            t2 = F.hflip(t2)
            t3 = F.hflip(t3)
            t4 = F.hflip(t4)
        return image, t1, t2, t3, t4


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, t1, t2, t3, t4):
        image = pad_if_smaller(image, self.size)
        t1 = pad_if_smaller(t1, self.size, fill=255)
        t2 = pad_if_smaller(t2, self.size, fill=255)
        t3 = pad_if_smaller(t3, self.size, fill=255)
        t4 = pad_if_smaller(t4, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        t1 = F.crop(t1, *crop_params)
        t2 = F.crop(t2, *crop_params)
        t3 = F.crop(t3, *crop_params)
        t4 = F.crop(t4, *crop_params)
        return image, t1, t2, t3, t4


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, t1, t2, t3, t4):
        image = F.center_crop(image, self.size)
        t1 = F.center_crop(t1, self.size)
        t2 = F.center_crop(t2, self.size)
        t3 = F.center_crop(t3, self.size)
        t4 = F.center_crop(t4, self.size)
        return image, t1, t2, t3, t4


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target