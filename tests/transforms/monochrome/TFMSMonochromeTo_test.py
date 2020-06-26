import pytest

from image.ImageBatch import ImageBatch
from transforms.monochrome.TFMSMonochromeTo import TFMSMonochromeTo


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
