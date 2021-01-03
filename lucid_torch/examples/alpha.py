import torch
from torchvision import models

from ..image import ImageBatch
from ..objectives import (FCNeuronObjective, MeanOpacityObjective,
                          TVRegularizerObjective)
from ..renderer import RendererBuilder
from ..transforms import presets
from ..transforms.fft import TFMSIFFT
from ..transforms.unit_space.TFMSTrainingToUnitSpace import \
    TFMSTrainingToUnitSpace


def alpha(device="cuda:0", numberOfFrames=500):
    model = models.resnet18(pretrained=True)

    fcneuron = FCNeuronObjective(lambda m: m.fc, neuron=234)
    alpha = MeanOpacityObjective(torch.nn.Sequential(
        TFMSIFFT(),
        TFMSTrainingToUnitSpace()
    ))
    lowfreq = TVRegularizerObjective(torch.nn.Sequential(
        TFMSIFFT(),
        TFMSTrainingToUnitSpace()
    ))
    objective = fcneuron * (1.0 - alpha) * (1.0 - lowfreq)

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
