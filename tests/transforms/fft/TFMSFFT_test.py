import pytest
from transforms.fft.TFMSFFT import TFMSFFT


class TestTFMSFFT:
    # region __init__
    def test_init_raises_error_on_invalid_parameters(self):
        with pytest.raises(TypeError):
            TFMSFFT("not a number")
    # endregion
