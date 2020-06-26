from torchvision import models
import torch

from lucid_torch.image.ImageBatch import ImageBatch
from lucid_torch.renderer.Renderer import RendererBuilder
from lucid_torch.objectives.image.MeanOpacityObjective import MeanOpacityObjective
from lucid_torch.objectives.neuron.FCNeuronObjective import FCNeuronObjective
from lucid_torch.transforms import presets
from lucid_torch.transforms.unit_space.TFMSTrainingToUnitSpace import TFMSTrainingToUnitSpace
from lucid_torch.transforms.fft.TFMSIFFT import TFMSIFFT


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
