from torchvision import models
import torch

from lucid_torch.image.ImageBatch import ImageBatch
from lucid_torch.renderer.Renderer import RendererBuilder
from lucid_torch.objectives.channel.ChannelObjective import ChannelObjective
from lucid_torch.transforms import presets


def basic(device="cuda:0", numberOfFrames=500):
    model = models.resnet18(pretrained=True)

    # look into 3rd layer, 10th channel; this changes a lot between architectures
    objective = ChannelObjective(lambda m: m.layer3[1].conv2, channel=15)

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
