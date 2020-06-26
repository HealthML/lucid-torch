from lucid_torch.image.ImageBatch import ImageBatch
from lucid_torch.transforms.unit_space.TFMSTrainingToUnitSpace import TFMSTrainingToUnitSpace


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
