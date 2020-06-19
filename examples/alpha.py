from torchvision import models
import torch

from image.ImageBatch import ImageBatch
from renderer.Renderer import RendererBuilder
from renderer.Renderer_internal import Renderer
from objectives import MeanOpacity, FCNeuron
from tfms import presets
from tfms.unit_space import TFMSTrainingToUnitSpace
from tfms.fft import TFMSIFFT


def alpha(device="cuda:0"):
    model = models.resnet18(pretrained=True)

    fcneuron = FCNeuron(lambda m: m.fc, neuron=234)
    alpha = MeanOpacity(torch.nn.Sequential(
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

    renderer: Renderer = (RendererBuilder()
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
    renderer.render(1000)
    return renderer.drawableImageBatch().data
