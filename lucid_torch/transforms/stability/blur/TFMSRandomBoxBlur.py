from random import randint

import torch

from lucid_torch.transforms.stability.blur.TFMSBoxBlur import TFMSBoxBlur


class TFMSRandomBoxBlur(torch.nn.Module):
    def __init__(self, min_kernel_size=(11, 11), max_kernel_size=(51, 51), border_type='reflect'):
        super(TFMSRandomBoxBlur, self).__init__()
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size
        self.border_type = border_type

    def forward(self, img):
        return TFMSBoxBlur((
            randint(self.min_kernel_size[0] // 2,
                    self.max_kernel_size[0] // 2) * 2 + 1,
            randint(self.min_kernel_size[1] // 2,
                    self.max_kernel_size[1] // 2) * 2 + 1
        ), border_type=self.border_type)(img)
