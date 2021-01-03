import torch
from torchvision import models

from ..image import ImageBatch
from ..objectives import (ChannelObjective, ConvNeuronObjective,
                          FCNeuronObjective)
from ..renderer import RendererBuilder


def objective(device="cuda:0", numberOfFrames=500):
    model = models.resnet18(pretrained=True)

    # visualize full channel
    obj1 = ChannelObjective(lambda m: m.layer3[1].conv2, channel=15)
    # visualize single (center) neuron in this channel
    obj2 = ConvNeuronObjective(lambda m: m.layer3[1].conv2, channel=15)
    # visualize neuron in fully connected layer, in this case already output
    obj3 = FCNeuronObjective(lambda m: m.fc, neuron=123)

    # objectives can be combined
    objective = 0.5 * obj1 + obj2 - obj3

    imageBatch = ImageBatch.generate().to(device)

    optimizer = torch.optim.Adam([imageBatch.data],
                                 lr=0.05,
                                 eps=1e-7,
                                 weight_decay=0.0)

    renderer = (RendererBuilder()
                .imageBatch(imageBatch)
                .model(model)
                .optimizer(optimizer)
                .objective(objective)
                .withLivePreview()
                .build()
                )
    renderer.render(numberOfFrames)
    return renderer.drawableImageBatch().data
