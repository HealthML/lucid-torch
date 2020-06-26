import torch
from image.spatial import rfft2d_freqs
import numpy as np


def fft_scale(h: int, w: int, decay_power: float):
    freqs = rfft2d_freqs(h, w)
    scale = np.sqrt(h * w) / np.maximum(freqs, 1. / max(h, w)) ** decay_power
    scale = torch.from_numpy(scale).float()
    scale = scale.view(1, 1, *scale.shape, 1)
    return scale


class TFMSFFT(torch.nn.Module):
    def __init__(self, decay_power: float = 1.0):
        super(TFMSFFT, self).__init__()
        if not isinstance(decay_power, (float, int)):
            raise TypeError()
        self.decay_power = decay_power
        self.h = None
        self.w = None
        self.scale = None

    def forward(self, data: torch.Tensor):
        return torch.rfft(data, signal_ndim=2) / self._updateScale(data)

    def _updateScale(self, data: torch.Tensor):
        h = data.shape[2]
        w = data.shape[3]
        if (self.h != h) or (self.w != w):
            self.h = h
            self.w = w
            self.scale = fft_scale(h, w, self.decay_power)
        self.scale = self.scale.to(data.device)
        return self.scale


class TFMSIFFT(torch.nn.Module):
    def __init__(self, decay_power: float = 1.0):
        super(TFMSIFFT, self).__init__()
        if not isinstance(decay_power, (float, int)):
            raise TypeError()
        self.decay_power = decay_power
        self.h = None
        self.w = None
        self.scale = None

    def forward(self, data: torch.Tensor):
        data = self._updateScale(data) * data
        return torch.irfft(data,
                           signal_ndim=2,
                           signal_sizes=(self.h, self.w))

    def _updateScale(self, data: torch.Tensor):
        h = data.shape[2]
        w = data.shape[3] * 2 - 2
        if (self.h != h) or (self.w != w):
            self.h = h
            self.w = w
            self.scale = fft_scale(h, w, self.decay_power)
        self.scale = self.scale.to(data.device)
        return self.scale
