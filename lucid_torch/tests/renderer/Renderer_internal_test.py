from ...renderer.Observer import RendererObserver
from ...renderer.Renderer import Renderer


# region fakes
class FakeObserver(RendererObserver):
    def __init__(self, renderer: Renderer):
        super(FakeObserver, self).__init__()
        self.rendering = False
        self.numFrameCallbacks = 0
        self.numFramesToRender = 0
        self.numStartRenderCalled = 0
        self.renderer = renderer

    def onStartRender(self, renderer: Renderer, numberOfFrames: int):
        assert isinstance(numberOfFrames, int)
        assert numberOfFrames >= 0
        assert self.renderer is renderer
        assert not self.rendering
        self.rendering = True
        self.numFrameCallbacks = 0
        self.numFramesToRender = numberOfFrames
        self.numStartRenderCalled = self.numStartRenderCalled + 1

    def onFrame(self, renderer: Renderer):
        assert self.renderer is renderer
        assert self.rendering
        self.numFrameCallbacks = self.numFrameCallbacks + 1

    def onStopRender(self, renderer: Renderer):
        assert self.renderer is renderer
        assert self.rendering
        assert self.numFrameCallbacks == self.numFramesToRender
        self.rendering = False
# endregion


class TestVideoExporter:
    # region render
    def test_renderCallsObservers(self, basicRenderer):
        observer = FakeObserver(basicRenderer)
        basicRenderer.add_observer(observer)
        basicRenderer.render(1)
        assert observer.numStartRenderCalled == 1
        assert not observer.rendering

    def test_renderTrainsImage(self, basicRenderer):
        data = basicRenderer.imageBatch.data.clone()
        basicRenderer.render(1)
        assert not data.equal(basicRenderer.imageBatch.data)
    # endregion

    # region remove_observer
    def test_removeObserverRemovesObserver(self, basicRenderer):
        observer = FakeObserver(basicRenderer)
        basicRenderer.add_observer(observer)
        basicRenderer.remove_observer(observer)
        basicRenderer.render(1)
        assert observer.numStartRenderCalled == 0
    # endregion
