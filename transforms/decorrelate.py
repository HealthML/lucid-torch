from image.color import linear_decorrelate
import torch


class TFMSDecorrelate(torch.nn.Module):
    def forward(self, data: torch.Tensor):
        return linear_decorrelate(data)
