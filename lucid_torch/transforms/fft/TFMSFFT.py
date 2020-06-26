import torch

from .scale import fft_scale


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
