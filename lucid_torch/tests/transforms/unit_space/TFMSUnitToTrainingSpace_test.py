from ....image import ImageBatch
from ....transforms.unit_space.TFMSUnitToTrainingSpace import \
    TFMSUnitToTrainingSpace


class TesteTFMSUnitToTrainingSpace:
    # region forward
    def test_forward_keeps_shape(self):
        imageBatch = ImageBatch.generate()
        transformed_imageBatch = imageBatch.transform(
            TFMSUnitToTrainingSpace())
        assert imageBatch.data.shape == transformed_imageBatch.data.shape
    # endregion
