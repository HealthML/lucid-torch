import numpy as np
from IPython import display
from matplotlib import pyplot as plt

from lucid_torch.renderer.Observer import RendererObserver
from lucid_torch.renderer.Renderer_internal import Renderer


class RendererLivePreview(RendererObserver):
    def __init__(self, numberSkipsBetweenUpdates: int = 10):
        super(RendererLivePreview, self).__init__()
        if not isinstance(numberSkipsBetweenUpdates, int):
            raise TypeError()
        elif numberSkipsBetweenUpdates < 0:
            raise ValueError()
        self.numberSkipsBetweenUpdates = numberSkipsBetweenUpdates
        self.skipped = numberSkipsBetweenUpdates

    def _draw(self, renderer: Renderer):
        imgs = renderer.drawableImageBatch()
        imgs = imgs.data.detach().cpu().numpy()
        imgs = np.moveaxis(imgs, 1, -1)
        n_img = imgs.shape[0]
        if n_img < 13:
            n_rows = [None, 1, 1, 2, 2, 2, 2,
                      3, 3, 3, 3, 3, 3][n_img]
        else:
            n_rows = 4
        n_cols = np.ceil(n_img / n_rows).astype(int)
        fig = plt.figure(figsize=(10, 10))
        display.clear_output(wait=False)
        for i, img in enumerate(imgs):
            fig.add_subplot(n_rows, n_cols, i + 1)
            plt.imshow(img)
        display.display(plt.gcf())

    def onStartRender(self, renderer: Renderer, numberOfFrames: int):
        self.skipped = self.numberSkipsBetweenUpdates

    def onStopRender(self, renderer: Renderer):
        self._draw(renderer)

    def onFrame(self, renderer: Renderer):
        if self.skipped >= self.numberSkipsBetweenUpdates:
            self.skipped = 0
            self._draw(renderer)
        else:
            self.skipped = self.skipped + 1
