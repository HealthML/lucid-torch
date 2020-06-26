import numpy as np
import torch


class TFMSJitter(torch.nn.Module):
    '''jitter in the lucid sense, not in the standard definition'''

    def __init__(self, d):
        super(TFMSJitter, self).__init__()
        self.d = d

    def forward(self, img):
        w, h = img.shape[-2:]
        w_start, h_start = np.random.choice(self.d, 2)
        w_end, h_end = w + w_start - self.d, h + h_start - self.d
        return img[:, :, w_start:w_end, h_start:h_end]
