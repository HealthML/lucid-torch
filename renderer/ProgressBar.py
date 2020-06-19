from tqdm import tqdm

from renderer.Observer import RendererObserver
from renderer.Renderer_internal import Renderer


class RendererProgressBar(RendererObserver):
    def __init__(self):
        super(RendererProgressBar, self).__init__()
        self.__pbar: tqdm = None
        self.__epoch = 0

    def onStartRender(self, renderer: Renderer, numberOfFrames: int):
        self.__pbar = tqdm(total=numberOfFrames)

    def onStopRender(self, renderer: Renderer):
        self.__pbar.close()

    def onFrame(self, renderer: Renderer):
        self.__epoch = self.__epoch + 1
        self.__pbar.update(1)
        self.__pbar.set_description(
            f"Epoch {self.__epoch}, current loss: {round(renderer.loss(), 3)}")
