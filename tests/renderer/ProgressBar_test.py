from lucid_torch.renderer.ProgressBar import RendererProgressBar


class TestProgressBar:
    def test_noErrorRaisedOnRendererCallbacks(self, basicRenderer):
        progressBar = RendererProgressBar()
        progressBar.onStartRender(basicRenderer, 1)
        progressBar.onFrame(basicRenderer)
        progressBar.onStopRender(basicRenderer)
