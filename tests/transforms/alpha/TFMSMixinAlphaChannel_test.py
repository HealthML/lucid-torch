import pytest
import torch

from lucid_torch.image.ImageBatch import ImageBatch
from lucid_torch.transforms.alpha.BackgroundStyle import BackgroundStyle
from lucid_torch.transforms.alpha.TFMSAddAlphaChannel import TFMSAddAlphaChannel
from lucid_torch.transforms.alpha.TFMSMixinAlphaChannel import TFMSMixinAlphaChannel


class TestTFMSMixinAlphaChannel:
    # region __init__
    def test_init_raises_error_on_invalid_parameters(self):
        with pytest.raises(TypeError):
            TFMSMixinAlphaChannel('not a valid background')
    # endregion

    # region forward
    def test_forward_removes_one_dimension(self):
        imageBatch = ImageBatch.generate(
            data_space_transform=TFMSAddAlphaChannel())
        num_channels = imageBatch.data.shape[1]
        transformed_imageBatch = imageBatch.transform(TFMSMixinAlphaChannel())
        transformed_num_channels = transformed_imageBatch.data.shape[1]
        assert num_channels - 1 == transformed_num_channels

    def test_forward_mixes_in_selected_color_if_transparent(self):
        def test_color(color: BackgroundStyle, value: float):
            imageBatch = ImageBatch.generate(
                data_space_transform=TFMSAddAlphaChannel())
            imageBatch.data[:, -1] = 0.0
            transformed_imageBatch = imageBatch.transform(
                TFMSMixinAlphaChannel(color))
            assert torch.all(
                (transformed_imageBatch.data - value).abs() <= 0.001)
        test_color(BackgroundStyle.WHITE, 1.0)
        test_color(BackgroundStyle.BLACK, 0.0)

    def test_forward_not_mixes_in_selected_color_if_opaque(self):
        def test_color(color: BackgroundStyle):
            imageBatch = ImageBatch.generate(
                data_space_transform=TFMSAddAlphaChannel())
            imageBatch.data[:, -1] = 1.0
            transformed_imageBatch = imageBatch.transform(
                TFMSMixinAlphaChannel(color))
            assert torch.all(
                (transformed_imageBatch.data - imageBatch.data[:, :-1]).abs() <= 0.001)
        test_color(BackgroundStyle.WHITE)
        test_color(BackgroundStyle.BLACK)
    # endregion forward
