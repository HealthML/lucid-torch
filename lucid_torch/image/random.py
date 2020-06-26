import numpy as np
import torch

from lucid_torch.image.color import linear_decorrelate
from lucid_torch.image.spatial import rfft2d_freqs


def rand_fft(shape, std=0.2, decay_power=1.5, decorrelate=True):
    b, ch, h, w = shape
    imgs = []
    for _ in range(b):
        freqs = rfft2d_freqs(h, w)
        fh, fw = freqs.shape
        spectrum_var = torch.normal(0, std, (3, fh, fw, 2))
        scale = np.sqrt(h * w) / np.maximum(freqs, 1. / max(h, w))**decay_power

        scaled_spectrum = spectrum_var * \
            torch.from_numpy(scale.reshape(1, *scale.shape, 1)).float()
        img = torch.irfft(scaled_spectrum, signal_ndim=2)
        img = img[:ch, :h, :w]
        imgs.append(img)

    # 4 for desaturation / better scale...
    imgs = torch.stack(imgs) / 4
    if decorrelate:
        imgs = linear_decorrelate(imgs)
    return imgs.sigmoid()
