import torch

from ....image import ImageBatch
from ....transforms.unit_space.TFMSTrainingToUnitSpace import \
    TFMSTrainingToUnitSpace
from ....transforms.unit_space.TFMSUnitToTrainingSpace import \
    TFMSUnitToTrainingSpace


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
