import pytest
import torch
from torchvision import models

from image.ImageBatch import ImageBatch
from objectives import Channel
from renderer.Renderer import RendererBuilder


@pytest.fixture
def imageBatch():
    return ImageBatch.generate()


@pytest.fixture
def model():
    return models.resnet18(pretrained=True)


@pytest.fixture
def optimizer(imageBatch):
    return torch.optim.Adam([imageBatch.data],
                            lr=0.05,
                            eps=1e-7,
                            weight_decay=0.0)


@pytest.fixture
def objective():
    return Channel(lambda m: m.layer3[1].conv2, channel=15)


@pytest.fixture
def trainTFMS():
    return torch.nn.Identity()


@pytest.fixture
def drawTFMS():
    return torch.nn.Identity()


@pytest.fixture
def basicRendererBuilder(imageBatch, model, optimizer, objective, trainTFMS, drawTFMS):
    return (RendererBuilder()
            .imageBatch(imageBatch)
            .model(model)
            .optimizer(optimizer)
            .objective(objective)
            .trainTFMS(trainTFMS)
            .drawTFMS(drawTFMS))


@pytest.fixture
def basicRenderer(basicRendererBuilder):
    return basicRendererBuilder.build()