import pytest

from lucid_torch.image import ImageBatch
from lucid_torch.transforms.monochrome import TFMSToMonochrome


class TestTFMSToMonochrome:
    # region __init__
    def test_init_raises_error_on_invalid_parameter(self):
        with pytest.raises(TypeError):
            TFMSToMonochrome('not a list')
    # endregion

    # region forward
    def test_forward_creates_one_dimension(self):
        imageBatch = ImageBatch.generate(
            channels=3,
            data_space_transform=TFMSToMonochrome([0.3, 0.3, 0.4]))
        assert imageBatch.data.shape[1] == 1
    # endregion
