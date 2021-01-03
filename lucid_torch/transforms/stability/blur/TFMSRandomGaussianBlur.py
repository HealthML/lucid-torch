from random import randint

from torch import nn

from ...stability.blur import TFMSGaussianBlur


class TFMSRandomGaussianBlur(nn.Module):
    def __init__(self, min_kernel_size=(11, 11), max_kernel_size=(51, 51), min_sigma=(3, 3), max_sigma=(21, 21), border_type='reflect'):
        super(TFMSRandomGaussianBlur, self).__init__()
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size
        self.border_type = border_type
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def forward(self, img):
        return TFMSGaussianBlur((
            randint(self.min_kernel_size[0] // 2,
                    self.max_kernel_size[0] // 2) * 2 + 1,
            randint(self.min_kernel_size[1] // 2,
                    self.max_kernel_size[1] // 2) * 2 + 1
        ), (
            randint(self.min_sigma[0],
                    self.max_sigma[0]),
            randint(self.min_sigma[1],
                    self.max_sigma[1])
        ), border_type=self.border_type)(img)
