import pytest

from lucid_torch.renderer.LivePreview import RendererLivePreview


class TestLivePreview:
    # region __init__
    def test_initRaisesErrorOnInvalidParameters(self):
        with pytest.raises(TypeError):
            RendererLivePreview(0.2)
        with pytest.raises(ValueError):
            RendererLivePreview(-2)
        with pytest.raises(TypeError):
            RendererLivePreview('not a uint')
    # endregion

    def test_noErrorRaisedOnRendererCallbacks(self, basicRenderer):
        livePreview = RendererLivePreview()
        livePreview.onStartRender(basicRenderer, 1)
        livePreview.onFrame(basicRenderer)
        livePreview.onStopRender(basicRenderer)
