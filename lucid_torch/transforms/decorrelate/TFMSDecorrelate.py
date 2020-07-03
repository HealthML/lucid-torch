import torch

from lucid_torch.image.color import linear_decorrelate


class TFMSDecorrelate(torch.nn.Module):
    def forward(self, data: torch.Tensor):
        return linear_decorrelate(data)
