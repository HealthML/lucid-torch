import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TFMSNormalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(TFMSNormalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, img):
        dev = img.device
        self.mean, self.std = self.mean.to(dev), self.std.to(dev)
        return (img - self.mean[..., None, None]) / self.std[..., None, None]


class TFMSRandomScale(nn.Module):
    def __init__(self, scales=None, rng=None, mode='bilinear'):
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


class TFMSRandomRotate(nn.Module):
    def __init__(self, angles=None, rng=None):
        super(TFMSRandomRotate, self).__init__()
        self.angles = angles
        if angles is not None:
            self.rotations = []
            for a in angles:
                self.rotations.append(TFMSRotateGPU(a))
        self.rng = rng
        if (angles is None) and (rng is None):
            raise ValueError

    def forward(self, img):
        if self.angles is not None:
            rot = np.random.choice(self.rotations)
        else:
            angle = self.rng[0] + \
                (self.rng[1] - self.rng[0]) * np.random.rand()
            rot = TFMSRotateGPU(angle)

        return rot(img)


class TFMSRotateGPU(nn.Module):
    def __init__(self, angle=0):
        super(TFMSRotateGPU, self).__init__()
        self.angle = torch.tensor([angle * np.pi / 180])
        self.rot_matrix = torch.tensor([[torch.cos(self.angle), torch.sin(self.angle)],
                                        [-torch.sin(self.angle), torch.cos(self.angle)]])
        self.w = self.h = None

    def _set_up(self, w, h, dev):
        self.rot_matrix = self.rot_matrix.to(dev)
        self.w, self.h = w, h
        xx, yy = torch.meshgrid(torch.arange(
            w, device=dev), torch.arange(h, device=dev))
        xx, yy = xx.contiguous().float(), yy.contiguous().float()
        xm, ym = (w + 1) / 2, (h + 1) / 2

        inds = torch.cat([(xx - xm).view(-1, 1), (yy - ym).view(-1, 1)], dim=1)
        inds = (self.rot_matrix @ inds.t()).round() + \
            torch.tensor([[xm, ym]], device=dev).t()

        inds[inds < 0] = 0.
        inds[0, :][inds[0, :] >= w] = w - 1.
        inds[1, :][inds[1, :] >= h] = h - 1.
        self.inds = inds.long()
        self.xx, self.yy = xx.long(), yy.long()

    def forward(self, img):
        w, h = img.shape[-2:]
        dev = img.device
        if not (self.w, self.h) == (w, h):
            self._set_up(w, h, dev)

        rot_img = torch.zeros_like(img)
        rot_img[:, :, self.xx.view(-1), self.yy.view(-1)
                ] = img[:, :, self.inds[0, :], self.inds[1, :]]

        return rot_img


class TFMSGaussianNoise(nn.Module):
    def __init__(self, level=0.01):
        super(TFMSGaussianNoise, self).__init__()
        self.level = level

    def forward(self, img):
        return img + self.level * torch.randn_like(img)


class TFMSBlur(nn.Module):
    def __init__(self, kernel=torch.ones(3, 3)):
        super(TFMSBlur, self).__init__()
        self.kernel = kernel.view(1, 1, *kernel.shape).repeat(3, 1, 1, 1)
        self.pad = kernel.shape[0] // 2

    def forward(self, img):
        self.kernel = self.kernel.to(img.device)
        return F.conv2d(img, weight=self.kernel, groups=3, padding=self.pad)


class TFMSGaussianBlur(TFMSBlur):
    def __init__(self, kernel_size=3, std=1):
        ax = torch.linspace(-(kernel_size - 1) / 2.,
                            (kernel_size - 1) / 2, kernel_size)
        xx, yy = torch.meshgrid(ax, ax)
        kernel = torch.exp(-0.5 * (xx**2 + yy**2) / std**2)
        super(TFMSGaussianBlur, self).__init__(kernel=kernel / kernel.sum())


class TFMSRotate(nn.Module):
    def __init__(self, angle=0):
        super(TFMSRotate, self).__init__()
        self.angle = torch.tensor([angle * np.pi / 180])
        self.rot_matrix = torch.tensor([[torch.cos(self.angle), torch.sin(self.angle)],
                                        [-torch.sin(self.angle), torch.cos(self.angle)]])

    def forward(self, img):
        w, h = img.shape[-2:]

        xx, yy = torch.meshgrid(torch.arange(w), torch.arange(h))
        xx, yy = xx.contiguous().float(), yy.contiguous().float()
        xm, ym = (w + 1) / 2, (h + 1) / 2

        inds = torch.cat([(xx - xm).view(-1, 1), (yy - ym).view(-1, 1)], dim=1)
        inds = (self.rot_matrix @ inds.t()).round() + \
            torch.tensor([[xm, ym]]).t()

        inds[inds < 0] = 0.
        inds[0, :][inds[0, :] >= w] = w - 1.
        inds[1, :][inds[1, :] >= h] = h - 1.
        inds = inds.long()

        rot_img = torch.zeros_like(img)
        rot_img[:, :, xx.view(-1).long(), yy.view(-1).long()
                ] = img[:, :, inds[0, :], inds[1, :]]

        return rot_img
