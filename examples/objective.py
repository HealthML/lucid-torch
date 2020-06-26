from torchvision import models
import torch

from image.ImageBatch import ImageBatch
from renderer.Renderer import RendererBuilder
from objectives.channel.ChannelObjective import ChannelObjective
from objectives.neuron.ConvNeuronObjective import ConvNeuronObjective
from objectives.neuron.FCNeuronObjective import FCNeuronObjective
from tfms import presets


def objective(device="cuda:0", numberOfFrames=1):
    model = models.resnet18(pretrained=True)

    # visualize full channel
    obj1 = ChannelObjective(lambda m: m.layer3[1].conv2, channel=15)
    # visualize single (center) neuron in this channel
    obj2 = ConvNeuronObjective(lambda m: m.layer3[1].conv2, channel=15)
    # visualize neuron in fully connected layer, in this case already output
    obj3 = FCNeuronObjective(lambda m: m.fc, neuron=123)

    # objectives can be combined
    objective = 0.5 * obj1 + obj2 - obj3

    # return 4 visualizations of size 224 x 224
    imageBatch = ImageBatch.generate(
        width=224,
        height=224,
        batch_size=4,
        channels=3,
        data_space_transform=presets.dataspaceTFMS()
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
                .trainTFMS(presets.trainTFMS())
                .drawTFMS(presets.drawTFMS())
                .withLivePreview()
                .withProgressBar()
                .build()
                )
    renderer.render(numberOfFrames)
    return renderer.drawableImageBatch().data
