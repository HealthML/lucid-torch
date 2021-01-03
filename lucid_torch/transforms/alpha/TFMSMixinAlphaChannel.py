import torch

from ...image.random import rand_fft
from .BackgroundStyle import BackgroundStyle


class TFMSMixinAlphaChannel(torch.nn.Module):
    def __init__(self, background: BackgroundStyle = BackgroundStyle.RAND_FFT):
        super(TFMSMixinAlphaChannel, self).__init__()
        if not isinstance(background, BackgroundStyle):
            raise TypeError()
        self.background = background

    def forward(self, data: torch.Tensor):
        img = data[:, :-1]
        alpha = data[:, -1:]
        if self.background is BackgroundStyle.BLACK:
            background = torch.zeros_like(img)
        elif self.background is BackgroundStyle.WHITE:
            background = torch.ones_like(img)
        elif self.background is BackgroundStyle.RAND:
            background = torch.rand_like(img)
        elif self.background is BackgroundStyle.RAND_FFT:
            background = rand_fft(img.shape).to(data.device)
        else:
            raise NotImplementedError
        img = img * alpha + background * (1. - alpha)
        return img
