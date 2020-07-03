from tqdm import tqdm

from .Observer import RendererObserver
from .Renderer import Renderer


class RendererProgressBar(RendererObserver):
    def __init__(self):
        super(RendererProgressBar, self).__init__()
        self._pbar: tqdm = None
        self._epoch = 0

    def onStartRender(self, renderer: Renderer, numberOfFrames: int):
        self._pbar = tqdm(total=numberOfFrames)

    def onStopRender(self, renderer: Renderer):
        self._pbar.close()

    def onFrame(self, renderer: Renderer):
        self._epoch = self._epoch + 1
        self._pbar.update(1)
        self._pbar.set_description(
            f"Epoch {self._epoch}, current loss: {round(renderer.loss(), 3)}")
