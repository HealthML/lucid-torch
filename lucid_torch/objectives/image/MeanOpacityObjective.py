import torch

from .ImageObjective import ImageObjective


class MeanOpacityObjective(ImageObjective):
    def __init__(self, unit_space_transforms: torch.nn.Module):
        super().__init__()
        self.unit_space_transforms = unit_space_transforms
        self._weights = torch.ones(1)

    def weights(self, like):
        if self._weights.shape != like.shape:
            self._weights = torch.ones(*like.shape)
            for i1 in range(like.shape[0]):
                for i2 in range(like.shape[1]):
                    w1 = i1 / like.shape[0]
                    w2 = i2 / like.shape[1]
                    self._weights[i1, i2] = (0.5 - w1) * (0.5 - w1) + (0.5 - w2) * (0.5 - w2)
            self._weights = self._weights.to(like.device).sqrt()
        return self._weights

    def _compute_loss(self, imageBatch):
        self.imageBatch = imageBatch
        unmodified = self.imageBatch.unmodified()
        opacity = unmodified.transform(self.unit_space_transforms).data[:, -1]
        return (opacity * self.weights(opacity[0])).mean()
