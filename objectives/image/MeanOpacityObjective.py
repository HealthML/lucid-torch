import torch

from objectives.image.ImageObjective import ImageObjective


class MeanOpacityObjective(ImageObjective):
    def __init__(self, unit_space_tfms: torch.nn.Module):
        super().__init__()
        self.unit_space_tfms = unit_space_tfms

    def _compute_loss(self, imageBatch):
        self.imageBatch = imageBatch
        unmodified = self.imageBatch.unmodified()
        opacity = unmodified.transform(self.unit_space_tfms).data[-1]
        return opacity.sigmoid().mean()
