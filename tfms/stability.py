from random import randint
from typing import Tuple, Union

import kornia
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TFMSNormalize(kornia.augmentation.Normalize):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(TFMSNormalize, self).__init__(
            torch.tensor(mean),
            torch.tensor(std))


class TFMSRandomScale(nn.Module):
    def __init__(self, scales: Union[Tuple[float], None] = None, rng=None, mode='bilinear'):
        super(TFMSRandomScale, self).__init__()
        self.scales = scales
        self.rng = rng
        if (scales is None) and (rng is None):
            raise ValueError
        self.mode = mode

    def forward(self, img):
        if self.scales is not None:
            scale = np.random.choice(self.scales)
        else:
            scale = self.rng[0] + \
                (self.rng[1] - self.rng[0]) * np.random.rand()
        return F.interpolate(img, scale_factor=scale, mode=self.mode, align_corners=False)


class TFMSJitter(nn.Module):
    '''jitter in the lucid sense, not in the standard definition'''

    def __init__(self, d):
        super(TFMSJitter, self).__init__()
        self.d = d

    def forward(self, img):
        w, h = img.shape[-2:]
        w_start, h_start = np.random.choice(self.d, 2)
        w_end, h_end = w + w_start - self.d, h + h_start - self.d
        return img[:, :, w_start:w_end, h_start:h_end]


class TFMSPad(nn.Module):
    def __init__(self, w, mode='constant', constant_value=0.5):
        super(TFMSPad, self).__init__()
        self.w = w
        self.mode = mode
        self.constant_value = constant_value

    def forward(self, img):
        if self.constant_value == 'uniform':
            val = torch.rand(1)
        else:
            val = self.constant_value
        pad = nn.ConstantPad2d(self.w, val)
        return pad(img)


# Kornia has a random rotate class but unfortinately its padding mode is set to 'zeros' and cannot be changed
# TODO: Feature request in Kornia
class TFMSRandomRotate(nn.Module):
    def __init__(self, angles=None, rng=None):
        super(TFMSRandomRotate, self).__init__()
        self.angles = angles
        if angles is not None:
            self.rotations = []
            for a in angles:
                self.rotations.append(TFMSRotate(a))
        self.rng = rng
        if (angles is None) and (rng is None):
            raise ValueError

    def forward(self, img):
        if self.angles is not None:
            rot = np.random.choice(self.rotations)
        else:
            angle = self.rng[0] + \
                (self.rng[1] - self.rng[0]) * np.random.rand()
            rot = TFMSRotate(angle)
        return rot(img)


class TFMSRotate(nn.Module):
    def __init__(self, angle=0, padding_mode='border', interpolation='nearest'):
        super(TFMSRotate, self).__init__()
        self.angle = torch.tensor([angle])
        self.padding_mode = padding_mode
        self.interpolation = interpolation

    def forward(self, img: torch.Tensor):
        b, c, h, w = img.shape
        center = torch.tensor([[w, h]], dtype=torch.float) / 2
        transformation_matrix = kornia.get_rotation_matrix2d(
            center,
            self.angle,
            torch.ones(1))
        transformation_matrix = transformation_matrix.expand(
            b, -1, -1)
        transformation_matrix = transformation_matrix.to(img.device)
        return kornia.warp_affine(
            img.float(),
            transformation_matrix,
            dsize=(h, w),
            flags=self.interpolation,
            padding_mode=self.padding_mode)


class TFMSGaussianNoise(nn.Module):
    def __init__(self, level=0.01):
        super(TFMSGaussianNoise, self).__init__()
        self.level = level

    def forward(self, img):
        return img + self.level * torch.randn_like(img)


TFMSBoxBlur = kornia.filters.BoxBlur
TFMSMedianBlur = kornia.filters.MedianBlur
TFMSGaussianBlur = kornia.filters.GaussianBlur2d


class TFMSRandomBoxBlur(nn.Module):
    def __init__(self, min_kernel_size=(11, 11), max_kernel_size=(51, 51), border_type='reflect'):
        super(TFMSRandomBoxBlur, self).__init__()
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size
        self.border_type = border_type

    def forward(self, img):
        return TFMSBoxBlur((
            randint(self.min_kernel_size[0] // 2,
                    self.max_kernel_size[0] // 2) * 2 + 1,
            randint(self.min_kernel_size[1] // 2,
                    self.max_kernel_size[1] // 2) * 2 + 1
        ), border_type=self.border_type)(img)


class TFMSRandomGaussBlur(nn.Module):
    def __init__(self, min_kernel_size=(11, 11), max_kernel_size=(51, 51), min_sigma=(3, 3), max_sigma=(21, 21), border_type='reflect'):
        super(TFMSRandomGaussBlur, self).__init__()
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size
        self.border_type = border_type
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def forward(self, img):
        return TFMSGaussianBlur((
            randint(self.min_kernel_size[0] // 2,
                    self.max_kernel_size[0] // 2) * 2 + 1,
            randint(self.min_kernel_size[1] // 2,
                    self.max_kernel_size[1] // 2) * 2 + 1
        ), (
            randint(self.min_sigma[0],
                    self.max_sigma[0]),
            randint(self.min_sigma[1],
                    self.max_sigma[1])
        ), border_type=self.border_type)(img)
