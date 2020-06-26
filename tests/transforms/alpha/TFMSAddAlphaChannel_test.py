import pytest

from lucid_torch.image.ImageBatch import ImageBatch
from lucid_torch.transforms.alpha.TFMSAddAlphaChannel import TFMSAddAlphaChannel


class TestTFMSAddAlphaChannel:
    # region __init__
    def test_init_raises_error_on_invalid_parameters(self):
        with pytest.raises(TypeError):
            TFMSAddAlphaChannel(mean='not a number')
        with pytest.raises(TypeError):
            TFMSAddAlphaChannel(std='not a number')
        with pytest.raises(TypeError):
            TFMSAddAlphaChannel(unit_space='not a bool')
    # endregion

    # region forward
    def test_forward_adds_channel(self):
        imageBatch = ImageBatch.generate()
        num_channels = imageBatch.data.shape[1]
        transformed_imageBatch = imageBatch.transform(TFMSAddAlphaChannel())
        transformed_num_channels = transformed_imageBatch.data.shape[1]
        assert num_channels + 1 == transformed_num_channels
    # endregion
