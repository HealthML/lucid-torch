from torchvision import models

from ..image import ImageBatch
from ..objectives import ChannelObjective
from ..renderer import RendererBuilder


def basic(device="cuda:0", numberOfFrames=500):
    model = models.resnet18(pretrained=True)

    # look into 3rd layer, 10th channel; this changes a lot between architectures
    objective = ChannelObjective(lambda m: m.layer3[1].conv2, channel=15)

    # return 4 visualizations of size 224 x 224
    imageBatch = ImageBatch.generate(
        width=224,
        height=224,
        batch_size=4,
        channels=3
    ).to(device)

    renderer = (RendererBuilder()
                .imageBatch(imageBatch)
                .model(model)
                .objective(objective)
                .withLivePreview()
                .build()
                )
    renderer.render(numberOfFrames)
    return renderer.drawableImageBatch().data
