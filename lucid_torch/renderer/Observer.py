from abc import ABC, abstractmethod


class RendererObserver(ABC):
    @abstractmethod
    def onStartRender(self, renderer, numberOfFrames: int):
        pass

    @abstractmethod
    def onStopRender(self, renderer):
        pass

    @abstractmethod
    def onFrame(self, renderer):
        pass
