import torch

from ..image import ImageBatch
from ..objectives import Objective
from ..utils import prep_model
from .Observer import RendererObserver


class Renderer:
    def __init__(self,
                 imageBatch: ImageBatch,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 objective: Objective,
                 trainTFMS: torch.nn.Module,
                 drawTFMS: torch.nn.Module):
        self.imageBatch = imageBatch
        self.model = prep_model(model, self.imageBatch.data.device)
        self.optimizer = optimizer
        self.objective = objective
        self.objective.register(self.model)
        self.trainTFMS = trainTFMS
        self.drawTFMS = drawTFMS
        self.observers = []
        self._loss = 0

    def __del__(self):
        self.objective.remove_hook()

    def add_observer(self, observer: RendererObserver):
        if not isinstance(observer, RendererObserver):
            raise TypeError()
        self.observers.append(observer)
        return self

    def remove_observer(self, observer: RendererObserver):
        if not isinstance(observer, RendererObserver):
            raise TypeError()
        self.observers.remove(observer)
        return self

    def render(self, numberOfFrames: int):
        if not isinstance(numberOfFrames, int):
            raise TypeError()
        if numberOfFrames < 1:
            raise ValueError()

        self._startRender(numberOfFrames)
        for _ in range(numberOfFrames):
            self._frame()
        self._stopRender()
        return self

    def drawableImageBatch(self):
        drawable = self.imageBatch.transform(self.drawTFMS)
        return drawable

    def loss(self):
        return self._loss

    def _startRender(self, numberOfFrames: int):
        for observer in self.observers:
            observer.onStartRender(self, numberOfFrames)

    def _frame(self):
        self.optimizer.zero_grad()
        transformedImageBatch = self.imageBatch.transform(self.trainTFMS)
        self.model(transformedImageBatch.data)
        self._loss = self.objective.backward(transformedImageBatch)
        self.optimizer.step()
        for observer in self.observers:
            observer.onFrame(self)

    def _stopRender(self):
        for observer in self.observers:
            observer.onStopRender(self)
