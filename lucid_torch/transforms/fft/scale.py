import numpy as np
import torch

from lucid_torch.image.spatial import rfft2d_freqs


def fft_scale(h: int, w: int, decay_power: float):
    freqs = rfft2d_freqs(h, w)
    scale = np.sqrt(h * w) / np.maximum(freqs, 1. / max(h, w)) ** decay_power
    scale = torch.from_numpy(scale).float()
    scale = scale.view(1, 1, *scale.shape, 1)
    return scale
