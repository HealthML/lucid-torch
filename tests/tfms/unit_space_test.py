import torch

from image.ImageBatch import ImageBatch
from tfms.unit_space import TFMSTrainingToUnitSpace, TFMSUnitToTrainingSpace


class TestTFMSTrainingToUnitSpace:
    # region forward
    def test_forward_keeps_shape(self):
        imageBatch = ImageBatch.generate()
        transformed_imageBatch = imageBatch.transform(
            TFMSTrainingToUnitSpace())
        assert imageBatch.data.shape == transformed_imageBatch.data.shape

    def test_forward_converts_to_unit_space(self):
        imageBatch = ImageBatch([[[[100.0]]]])
        transformed_imageBatch = imageBatch.transform(
            TFMSTrainingToUnitSpace())
        assert transformed_imageBatch.data.item() >= 0.0
        assert transformed_imageBatch.data.item() <= 1.0
        # endregion


class TesteTFMSUnitToTrainingSpace:
    # region forward
    def test_forward_keeps_shape(self):
        imageBatch = ImageBatch.generate()
        transformed_imageBatch = imageBatch.transform(
            TFMSUnitToTrainingSpace())
        assert imageBatch.data.shape == transformed_imageBatch.data.shape
    # endregion


def test_unit_space_transforms_invert_each_other():
    imageBatch = ImageBatch.generate()
    training = imageBatch.transform(
        TFMSUnitToTrainingSpace())
    unit = training.transform(TFMSTrainingToUnitSpace())
    assert torch.all((imageBatch.data - unit.data).abs() <= 0.05)
    unit = imageBatch.transform(TFMSTrainingToUnitSpace())
    training = unit.transform(
        TFMSUnitToTrainingSpace())
    assert torch.all((imageBatch.data - training.data).abs() <= 0.05)
