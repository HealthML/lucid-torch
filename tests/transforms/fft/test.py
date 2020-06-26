import torch

from image.ImageBatch import ImageBatch
from transforms.fft.TFMSFFT import TFMSFFT
from transforms.fft.TFMSIFFT import TFMSIFFT


def test_fft_transforms_invert_each_other():
    imageBatch = ImageBatch.generate()
    fft = imageBatch.transform(TFMSFFT())
    rgb = fft.transform(TFMSIFFT())
    assert torch.all((imageBatch.data - rgb.data).abs() <= 0.001)
