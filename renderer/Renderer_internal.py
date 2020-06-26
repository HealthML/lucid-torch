import torch

from image.ImageBatch import ImageBatch
from objectives.Objective import Objective
from renderer.Observer import RendererObserver
from utils import prep_model


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
        self.__loss = 0

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

        self.__startRender(numberOfFrames)
        for _ in range(numberOfFrames):
            self.__frame()
        self.__stopRender()
        return self

    def drawableImageBatch(self):
        drawable = self.imageBatch.transform(self.drawTFMS)
        return drawable

    def loss(self):
        return self.__loss

    def __startRender(self, numberOfFrames: int):
        for observer in self.observers:
            observer.onStartRender(self, numberOfFrames)

    def __frame(self):
        self.optimizer.zero_grad()
        transformedImageBatch = self.imageBatch.transform(self.trainTFMS)
        self.model(transformedImageBatch.data)
        self.__loss = self.objective.backward(transformedImageBatch)
        self.optimizer.step()
        for observer in self.observers:
            observer.onFrame(self)

    def __stopRender(self):
        for observer in self.observers:
            observer.onStopRender(self)
