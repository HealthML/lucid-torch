import torch

from .ImageObjective import ImageObjective


class MeanOpacityObjective(ImageObjective):
    def __init__(self, unit_space_transforms: torch.nn.Module):
        super().__init__()
        self.unit_space_transforms = unit_space_transforms

    def _compute_loss(self, imageBatch):
        self.imageBatch = imageBatch
        unmodified = self.imageBatch.unmodified()
        opacity = unmodified.transform(self.unit_space_transforms).data[:, -1]
        return opacity.mean()
