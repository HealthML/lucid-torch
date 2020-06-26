from lucid_torch.image.ImageBatch import ImageBatch
from lucid_torch.transforms.channels.TFMSTransformChannels import TFMSTransformChannels
import pytest
import torch


class TFMSPlusTwo(torch.nn.Module):
    def forward(self, data):
        return data + 2


class Test_TFMSTransformChannels:
    # region __init__
    def test_init_raises_error_on_invalid_parameters(self):
        with pytest.raises(TypeError):
            TFMSTransformChannels(0.4, torch.nn.Identity())
        with pytest.raises(TypeError):
            TFMSTransformChannels("not an int", torch.nn.Identity())
        with pytest.raises(TypeError):
            TFMSTransformChannels(2, "not a transform")
    # endregion

    # region forward
    def test_forward_changes_multiple_selected_channels(self):
        imageBatch = ImageBatch.generate()
        transforms = TFMSTransformChannels([0, 2], TFMSPlusTwo())
        transformed_imageBatch = imageBatch.transform(transforms)
        expected_d0: torch.Tensor = imageBatch.data[:, 0] + 2
        real_d0: torch.Tensor = transformed_imageBatch.data[:, 0]
        assert expected_d0.equal(real_d0)
        expected_d2: torch.Tensor = imageBatch.data[:, 2] + 2
        real_d2: torch.Tensor = transformed_imageBatch.data[:, 2]
        assert expected_d2.equal(real_d2)

    def test_forward_changes_single_selected_channel(self):
        imageBatch = ImageBatch.generate()
        transformed_imageBatch = imageBatch.transform(
            TFMSTransformChannels(1, TFMSPlusTwo()))
        expected_d1: torch.Tensor = imageBatch.data[:, 1] + 2
        real_d1: torch.Tensor = transformed_imageBatch.data[:, 1]
        assert expected_d1.equal(real_d1)

    def test_forward_does_not_change_multiple_unselected_channels(self):
        imageBatch = ImageBatch.generate()
        transformed_imageBatch = imageBatch.transform(
            TFMSTransformChannels(1, TFMSPlusTwo()))
        expected_d0: torch.Tensor = imageBatch.data[:, 0]
        real_d0: torch.Tensor = transformed_imageBatch.data[:, 0]
        assert expected_d0.equal(real_d0)
        expected_d2: torch.Tensor = imageBatch.data[:, 2]
        real_d2: torch.Tensor = transformed_imageBatch.data[:, 2]
        assert expected_d2.equal(real_d2)

    def test_forward_does_not_change_single_unselected_channel(self):
        imageBatch = ImageBatch.generate()
        transformed_imageBatch = imageBatch.transform(
            TFMSTransformChannels([0, 2], TFMSPlusTwo()))
        expected_d1: torch.Tensor = imageBatch.data[:, 1]
        real_d1: torch.Tensor = transformed_imageBatch.data[:, 1]
        assert expected_d1.equal(real_d1)
    # endregion
