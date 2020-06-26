import torch

from lucid_torch.image.ImageBatch import ImageBatch
from lucid_torch.transforms.unit_space.TFMSTrainingToUnitSpace import TFMSTrainingToUnitSpace
from lucid_torch.transforms.unit_space.TFMSUnitToTrainingSpace import TFMSUnitToTrainingSpace


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
