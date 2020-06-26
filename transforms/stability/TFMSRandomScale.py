from typing import Tuple, Union

import numpy as np
import torch


class TFMSRandomScale(torch.nn.Module):
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
        return torch.functional.F.interpolate(img, scale_factor=scale, mode=self.mode, align_corners=False, recompute_scale_factor=True)
