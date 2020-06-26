from typing import Union

import torch

import lucid_torch.transforms.presets as presets
from lucid_torch.image import ImageBatch
from lucid_torch.objectives import Objective

from .LivePreview import RendererLivePreview
from .ProgressBar import RendererProgressBar
from .Renderer import Renderer
from .VideoExporter import RendererVideoExporter


class RendererBuilder:
    def __init__(self):
        self._imageBatch = None
        self._model = None
        self._optimizer = None
        self._objective = None
        self._trainTFMS = presets.trainTFMS()
        self._drawTFMS = presets.drawTFMS()
        self._videoFileName = None
        self._fps = None
        self._progressBar = True
        self._livePreview = False
        self._numberSkipsBetweenUpdates = None

    def imageBatch(self, imageBatch: ImageBatch):
        if not isinstance(imageBatch, ImageBatch):
            raise TypeError()
        self._imageBatch = imageBatch
        return self

    def model(self, model: torch.nn.Module):
        if not isinstance(model, torch.nn.Module):
            raise TypeError()
        self._model = model
        return self

    def optimizer(self, optimizer: torch.optim.Optimizer):
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError()
        self._optimizer = optimizer
        return self

    def objective(self, objective: Objective):
        if not isinstance(objective, Objective):
            raise TypeError()
        self._objective = objective
        return self

    def trainTFMS(self, transforms: torch.nn.Module):
        if not isinstance(transforms, torch.nn.Module):
            raise TypeError()
        self._trainTFMS = transforms
        return self

    def drawTFMS(self, transforms: torch.nn.Module):
        if not isinstance(transforms, torch.nn.Module):
            raise TypeError()
        self._drawTFMS = transforms
        return self

    def exportVideo(self, filename: Union[None, str], fps=60):
        if not (isinstance(filename, (str, None))):
            raise TypeError()
        if not isinstance(fps, int):
            raise TypeError()
        elif fps < 1:
            raise ValueError()
        self._videoFileName = filename
        self._fps = fps
        return self

    def withProgressBar(self, progressBar: bool = True):
        if not isinstance(progressBar, bool):
            raise TypeError()
        self._progressBar = progressBar
        return self

    def withLivePreview(self, livePreview: bool = True, numberSkipsBetweenUpdates: int = 50):
        if not isinstance(livePreview, bool):
            raise TypeError()
        if not isinstance(numberSkipsBetweenUpdates, int):
            raise TypeError()
        elif numberSkipsBetweenUpdates < 0:
            raise ValueError()
        self ._livePreview = livePreview
        self._numberSkipsBetweenUpdates = numberSkipsBetweenUpdates
        return self

    def _assertAllRequiredAttributesPresent(self):
        if self._imageBatch is None:
            raise AttributeError()
        if self._model is None:
            raise AttributeError()
        if self._objective is None:
            raise AttributeError()

    def _get_optimizer(self):
        if self._optimizer is None:
            return torch.optim.Adam([self._imageBatch.data],
                                    lr=0.05,
                                    eps=1e-7,
                                    weight_decay=0.0)
        else:
            return self._optimizer

    def _createRenderer(self):
        return Renderer(self._imageBatch,
                        self._model,
                        self._get_optimizer(),
                        self._objective,
                        self._trainTFMS,
                        self._drawTFMS)

    def _applyOptionalAttributes(self, renderer: Renderer):
        if self._videoFileName is not None:
            renderer.add_observer(
                RendererVideoExporter(self._videoFileName,
                                      self._imageBatch.width,
                                      self._imageBatch.height,
                                      self._fps))
        if self._progressBar:
            renderer.add_observer(RendererProgressBar())
        if self._livePreview:
            renderer.add_observer(RendererLivePreview(
                self._numberSkipsBetweenUpdates))

    def build(self) -> Renderer:
        self._assertAllRequiredAttributesPresent()
        renderer = self._createRenderer()
        self._applyOptionalAttributes(renderer)
        return renderer
