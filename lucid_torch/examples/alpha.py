import torch
from torchvision import models

from ..image import ImageBatch
from ..objectives import (ChannelObjective, MeanOpacityObjective,
                          TVRegularizerObjective)
from ..renderer import RendererBuilder
from ..transforms import presets
from ..transforms.fft import TFMSIFFT
from ..transforms.unit_space.TFMSTrainingToUnitSpace import \
    TFMSTrainingToUnitSpace


def alpha(device="cuda:0", numberOfFrames=500):
    model = models.resnet18(pretrained=True)

    base_objective = ChannelObjective(lambda m: m.layer3[1].conv2, channel=15)
    alpha = MeanOpacityObjective(torch.nn.Sequential(
        TFMSIFFT(),
        TFMSTrainingToUnitSpace()
    ))
    lowfreq = TVRegularizerObjective(torch.nn.Sequential(
        TFMSIFFT(),
        TFMSTrainingToUnitSpace()
    ))
    objective = base_objective * (1.0 - alpha) * (1.0 - lowfreq)

    imageBatch = ImageBatch.generate(
        data_space_transform=presets.dataspaceTFMS(alpha=True)
    ).to(device)

    renderer = (RendererBuilder()
                .imageBatch(imageBatch)
                .model(model)
                .objective(objective)
                .trainTFMS(presets.trainTFMS(alpha=True))
                .drawTFMS(presets.drawTFMS(alpha=True))
                .withLivePreview()
                .build()
                )
    renderer.render(numberOfFrames)
    return renderer.drawableImageBatch().data
