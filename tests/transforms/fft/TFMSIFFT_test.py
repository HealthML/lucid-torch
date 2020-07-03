import pytest

from lucid_torch.transforms.fft import TFMSIFFT


class TestTFMSIFFT:
    # region __init__
    def test_init_raises_error_on_invalid_parameters(self):
        with pytest.raises(TypeError):
            TFMSIFFT("not a number")
    # endregion
