from torchvision import models
import torch

from image.ImageBatch import ImageBatch
from renderer.Renderer import RendererBuilder
from objectives.image.MeanOpacityObjective import MeanOpacityObjective
from objectives.neuron.FCNeuronObjective import FCNeuronObjective
from transforms import presets
from transforms.unit_space import TFMSTrainingToUnitSpace
from transforms.fft import TFMSIFFT


def alpha(device="cuda:0", numberOfFrames=500):
    model = models.resnet18(pretrained=True)

    fcneuron = FCNeuronObjective(lambda m: m.fc, neuron=234)
    alpha = MeanOpacityObjective(torch.nn.Sequential(
        TFMSIFFT(),
        TFMSTrainingToUnitSpace()
    ))
    objective = fcneuron * (1.0 - alpha)

    imageBatch = ImageBatch.generate(
        data_space_transform=presets.dataspaceTFMS(alpha=True)
    ).to(device)

    optimizer = torch.optim.Adam([imageBatch.data],
                                 lr=0.05,
                                 eps=1e-7,
                                 weight_decay=0.0)

    renderer = (RendererBuilder()
                .imageBatch(imageBatch)
                .model(model)
                .optimizer(optimizer)
                .objective(objective)
                .trainTFMS(presets.trainTFMS(alpha=True))
                .drawTFMS(presets.drawTFMS(alpha=True))
                .withLivePreview()
                .withProgressBar()
                .build()
                )
    renderer.render(numberOfFrames)
    return renderer.drawableImageBatch().data
