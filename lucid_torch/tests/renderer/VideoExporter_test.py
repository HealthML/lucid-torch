import pytest

from ...renderer.VideoExporter import RendererVideoExporter


class TestVideoExporter:
    # region __init__
    def test_initRaisesErrorOnInvalidParameters(self):
        with pytest.raises(TypeError):
            RendererVideoExporter(0.2, 224, 224, 60)
        with pytest.raises(TypeError):
            RendererVideoExporter('/path/to/a/file', 'not a number', 224, 60)
        with pytest.raises(TypeError):
            RendererVideoExporter('/path/to/a/file', 224, 'not a number', 60)
        with pytest.raises(TypeError):
            RendererVideoExporter('/path/to/a/file', 224, 224, 'not a number')
        with pytest.raises(ValueError):
            RendererVideoExporter('/path/to/a/file', -10, 224, 60)
        with pytest.raises(ValueError):
            RendererVideoExporter('/path/to/a/file', 224, -10, 60)
        with pytest.raises(ValueError):
            RendererVideoExporter('/path/to/a/file', 224, 224, -10)
    # endregion

    def test_createsVideoFile(self, basicRenderer, tmpdir):
        videoPath = tmpdir.join("test.mp4")
        videoExporter = RendererVideoExporter(str(videoPath), 224, 224, 30)
        videoExporter.onStartRender(basicRenderer, 1)
        videoExporter.onFrame(basicRenderer)
        videoExporter.onStopRender(basicRenderer)
        assert videoPath.check()
