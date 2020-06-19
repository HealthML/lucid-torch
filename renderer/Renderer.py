from typing import Union

import torch

from image.ImageBatch import ImageBatch
from objectives import Objective
from renderer.LivePreview import RendererLivePreview
from renderer.ProgressBar import RendererProgressBar
from renderer.Renderer_internal import Renderer
from renderer.VideoExporter import RendererVideoExporter


class RendererBuilder:
    def __init__(self):
        self.__imageBatch = None
        self.__model = None
        self.__optimizer = None
        self.__objective = None
        self.__trainTFMS = None
        self.__drawTFMS = None
        self.__videoFileName = None
        self.__fps = None
        self.__progressBar = None
        self.__livePreview = None
        self.__numberSkipsBetweenUpdates = None

    def imageBatch(self, imageBatch: ImageBatch):
        if not isinstance(imageBatch, ImageBatch):
            raise TypeError()
        self.__imageBatch = imageBatch
        return self

    def model(self, model: torch.nn.Module):
        if not isinstance(model, torch.nn.Module):
            raise TypeError()
        self.__model = model
        return self

    def optimizer(self, optimizer: torch.optim.Optimizer):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError()
        self.__optimizer = optimizer
        return self

    def objective(self, objective: Objective):
        if not isinstance(objective, Objective):
            raise TypeError()
        self.__objective = objective
        return self

    def trainTFMS(self, tfms: torch.nn.Module):
        if not isinstance(tfms, torch.nn.Module):
            raise TypeError()
        self.__trainTFMS = tfms
        return self

    def drawTFMS(self, tfms: torch.nn.Module):
        if not isinstance(tfms, torch.nn.Module):
            raise TypeError()
        self.__drawTFMS = tfms
        return self

    def exportVideo(self, filename: Union[None, str], fps=60):
        if not (isinstance(filename, (str, None))):
            raise TypeError()
        if not isinstance(fps, int):
            raise TypeError()
        elif fps < 1:
            raise ValueError()
        self.__videoFileName = filename
        self.__fps = fps
        return self

    def withProgressBar(self, progressBar: bool = True):
        if not isinstance(progressBar, bool):
            raise TypeError()
        self.__progressBar = progressBar
        return self

    def withLivePreview(self, livePreview: bool = True, numberSkipsBetweenUpdates: int = 50):
        if not isinstance(livePreview, bool):
            raise TypeError()
        if not isinstance(numberSkipsBetweenUpdates, int):
            raise TypeError()
        elif numberSkipsBetweenUpdates < 0:
            raise ValueError()
        self .__livePreview = livePreview
        self.__numberSkipsBetweenUpdates = numberSkipsBetweenUpdates
        return self

    def __assertAllRequiredAttributesPresent(self):
        if self.__imageBatch is None:
            raise AttributeError()
        if self.__model is None:
            raise AttributeError()
        if self.__optimizer is None:
            raise AttributeError()
        if self.__objective is None:
            raise AttributeError()
        if self.__trainTFMS is None:
            raise AttributeError()
        if self.__drawTFMS is None:
            raise AttributeError()

    def __createRenderer(self):
        return Renderer(self.__imageBatch,
                        self.__model,
                        self.__optimizer,
                        self.__objective,
                        self.__trainTFMS,
                        self.__drawTFMS)

    def __applyOptionalAttributes(self, renderer: Renderer):
        if self.__videoFileName is not None:
            renderer.add_observer(
                RendererVideoExporter(self.__videoFileName,
                                      self.__imageBatch.width,
                                      self.__imageBatch.height,
                                      self.__fps))

        if (self.__progressBar is not None) and self.__progressBar:
            renderer.add_observer(RendererProgressBar())

        if (self.__livePreview is not None):
            renderer.add_observer(RendererLivePreview(
                self.__numberSkipsBetweenUpdates))

    def build(self) -> Renderer:
        self.__assertAllRequiredAttributesPresent()
        renderer = self.__createRenderer()
        self.__applyOptionalAttributes(renderer)
        return renderer
