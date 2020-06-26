import numpy as np


def rfft2d_freqs(h, w):
    '''compute 2D spectrum frequencies (from lucid)'''
    fy = np.fft.fftfreq(h)[:, None]
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[:((w // 2) + 2)]
    else:
        fx = np.fft.fftfreq(w)[:((w // 2) + 1)]
    return np.sqrt(fx * fx + fy * fy)
