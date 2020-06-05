import pytest
from tfms.monochrome import TFMSToMonochrome, TFMSMonochromeTo
from image.ImageBatch import ImageBatch


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


class TestTFMSMonochromeTo:
    # region __init__
    def test_init_raises_error_on_invalid_parameter(self):
        with pytest.raises(TypeError):
            TFMSMonochromeTo('not an integer')
        with pytest.raises(ValueError):
            TFMSMonochromeTo(0)
        with pytest.raises(ValueError):
            TFMSMonochromeTo(-12)
        with pytest.raises(ValueError):
            TFMSMonochromeTo(1)
        with pytest.raises(TypeError):
            TFMSMonochromeTo(0.23)
    # endregion

    # region forward
    def test_forward_creates_n_dimensions(self):
        imageBatch = ImageBatch.generate(channels=1)
        assert imageBatch.transform(TFMSMonochromeTo(3)).data.shape[1] == 3

    def test_forward_values_in_expaneded_dimensions_are_equal(self):
        imageBatch = ImageBatch.generate(channels=1)
        rgb = imageBatch.transform(TFMSMonochromeTo(3)).data
        for index in range(rgb.shape[1]):
            assert rgb[:, index].equal(rgb[:, 0])
    # endregion
