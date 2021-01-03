import numpy as np
import torch

from .TFMSRotate import TFMSRotate


class TFMSRandomRotate(torch.nn.Module):
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
