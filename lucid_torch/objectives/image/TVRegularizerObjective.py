import torch

from .ImageObjective import ImageObjective


class TVRegularizerObjective(ImageObjective):
    def __init__(self, unit_space_transforms: torch.nn.Module, alpha=1., update_thres=None, update_mult=None):
        super().__init__()
        self.alpha = alpha
        self.step = 0
        self.update_thres = update_thres
        self.update_mult = update_mult
        self.unit_space_transforms = unit_space_transforms

    def _compute_loss(self, imageBatch):
        imgs = imageBatch.unmodified().transform(self.unit_space_transforms).data
        tv_x = (imgs[..., :-1, :] - imgs[..., 1:, :]).abs().sum()
        tv_y = (imgs[..., :-1] - imgs[..., 1:]).abs().sum()
        loss = self.alpha * (tv_x + tv_y) / imgs.numel() / 2.0
        self._update()
        return loss

    def _update(self):
        self.step += 1
        if self.update_thres and (self.step in self.update_thres):
            self.alpha *= self.update_mult[self.update_thres.index(self.step)]
