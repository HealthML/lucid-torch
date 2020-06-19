import numpy as np
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

from renderer.Observer import RendererObserver
from renderer.Renderer_internal import Renderer


class RendererVideoExporter(RendererObserver):
    def __init__(self, filename: str, width: int, height: int, fps: float):
        super(RendererVideoExporter, self).__init__()
        if not isinstance(filename, str):
            raise TypeError()
        if not isinstance(width, int):
            raise TypeError()
        elif width < 1:
            raise ValueError()
        if not isinstance(height, int):
            raise TypeError()
        elif height < 1:
            raise ValueError()
        if not isinstance(fps, int):
            raise TypeError()
        elif fps < 1:
            raise ValueError()
        self.filename = filename
        self.width = width
        self.height = height
        self.fps = fps

    def onStartRender(self, renderer: Renderer, numberOfFrames: int):
        self.video = FFMPEG_VideoWriter(
            self.filename, (self.width, self.height), self.fps)

    def onStopRender(self, renderer: Renderer):
        self.video.close()

    def onFrame(self, renderer: Renderer):
        drawData = renderer.drawableImageBatch().data
        drawData = drawData[0]
        drawData = drawData.data.detach().cpu().numpy()
        drawData = np.moveaxis(drawData, 0, -1)
        drawData = np.uint8(255.0 * drawData)
        self.video.write_frame(drawData)
