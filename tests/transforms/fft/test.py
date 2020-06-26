import torch

from lucid_torch.image.ImageBatch import ImageBatch
from lucid_torch.transforms.fft.TFMSFFT import TFMSFFT
from lucid_torch.transforms.fft.TFMSIFFT import TFMSIFFT


def test_fft_transforms_invert_each_other():
    imageBatch = ImageBatch.generate()
    fft = imageBatch.transform(TFMSFFT())
    rgb = fft.transform(TFMSIFFT())
    assert torch.all((imageBatch.data - rgb.data).abs() <= 0.001)
