from enum import Enum, auto
import torch
from image.random import rand_fft


class TFMSAddAlphaChannel(torch.nn.Module):
    def __init__(self, mean: float = 0.5, std: float = 0.01, unit_space: bool = True):
        super(TFMSAddAlphaChannel, self).__init__()
        if not isinstance(std, (int, float)):
            raise TypeError()
        if not isinstance(mean, (int, float)):
            raise TypeError()
        if not isinstance(unit_space, bool):
            raise TypeError()
        self.std = std
        self.mean = mean
        self.unit_space = unit_space

    def forward(self, data: torch.Tensor):
        alpha_shape = (data.shape[0], 1, *data.shape[2:])
        alpha_channel = (torch
                         .normal(self.mean, self.std, alpha_shape)
                         .clamp(0.000001, 0.999999)
                         .to(data.device)
                         .requires_grad_(data.requires_grad))
        if not self.unit_space:
            alpha_channel = torch.log(alpha_channel / (1. - alpha_channel))
        return torch.cat([data, alpha_channel], 1)


class BackgroundStyle(Enum):
    BLACK = auto()
    WHITE = auto()
    RAND = auto()
    RAND_FFT = auto()


class TFMSMixinAlphaChannel(torch.nn.Module):
    def __init__(self, background: BackgroundStyle = BackgroundStyle.RAND_FFT):
        super(TFMSMixinAlphaChannel, self).__init__()
        if not isinstance(background, BackgroundStyle):
            raise TypeError()
        self.background = background

    def forward(self, data: torch.Tensor):
        img = data[:, :-1]
        alpha = data[:, -1]
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
