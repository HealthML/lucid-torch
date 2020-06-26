import numpy as np
import torch

from lucid_torch.transforms.stability.rotate.TFMSRotate import TFMSRotate

# Kornia has a random rotate class but unfortinately its padding mode is set to 'zeros' and cannot be changed
# TODO: Feature request in Kornia


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
