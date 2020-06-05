import pytest
import torch
from transforms.fft import TFMSFFT, TFMSIFFT
from image.ImageBatch import ImageBatch


class TestFFT:
    # region __init__
    def test_init_raises_error_on_invalid_parameters(self):
        with pytest.raises(TypeError):
            TFMSFFT("not a number")
    # endregion


class TestIFFT:
    # region __init__
    def test_init_raises_error_on_invalid_parameters(self):
        with pytest.raises(TypeError):
            TFMSIFFT("not a number")
    # endregion


def test_fft_transforms_invert_each_other():
    imageBatch = ImageBatch.generate()
    fft = imageBatch.transform(TFMSFFT())
    rgb = fft.transform(TFMSIFFT())
    assert torch.all((imageBatch.data - rgb.data).abs() <= 0.001)
