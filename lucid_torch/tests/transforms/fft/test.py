import torch

from ....image import ImageBatch
from ....transforms.fft import TFMSFFT, TFMSIFFT


def test_fft_transforms_invert_each_other():
    imageBatch = ImageBatch.generate()
    fft = imageBatch.transform(TFMSFFT())
    rgb = fft.transform(TFMSIFFT())
    assert torch.all((imageBatch.data - rgb.data).abs() <= 0.001)
