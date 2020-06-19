import pytest
import torch

from image.ImageBatch import ImageBatch
from objectives import ConstObjective
from renderer.LivePreview import RendererLivePreview
from renderer.ProgressBar import RendererProgressBar
from renderer.Renderer import RendererBuilder
from renderer.VideoExporter import RendererVideoExporter


class TestRendererBuilder:
    # region imageBatch
    def test_imageBatchRaisesErrorOnInvalidParameter(self):
        with pytest.raises(TypeError):
            RendererBuilder().imageBatch('not an image batch')
        with pytest.raises(TypeError):
            RendererBuilder().imageBatch(32)
    # endregion

    # region model
    def test_modelRaisesErrorOnInvalidParameter(self):
        with pytest.raises(TypeError):
            RendererBuilder().model('not a model')
        with pytest.raises(TypeError):
            RendererBuilder().model(32)
    # endregion

    # region optimizer
    def test_optimizerRaisesErrorOnInvalidParameter(self):
        with pytest.raises(TypeError):
            RendererBuilder().optimizer('not an optimizer')
        with pytest.raises(TypeError):
            RendererBuilder().optimizer(32)
    # endregion

    # region objective
    def test_objectivveRaisesErrorOnInvalidParameter(self):
        with pytest.raises(TypeError):
            RendererBuilder().objective('not an objective')
        with pytest.raises(TypeError):
            RendererBuilder().objective(32)
    # endregion

    # region trainTFMS
    def test_trainTFMSRaisesErrorOnInvalidParameter(self):
        with pytest.raises(TypeError):
            RendererBuilder().trainTFMS('not a module')
        with pytest.raises(TypeError):
            RendererBuilder().trainTFMS(32)
    # endregion

    # region drawTFMS
    def test_drawTFMSRaisesErrorOnInvalidParameter(self):
        with pytest.raises(TypeError):
            RendererBuilder().drawTFMS('not a module')
        with pytest.raises(TypeError):
            RendererBuilder().drawTFMS(32)
    # endregion

    # region exportVideo
    def test_exportVideoRaisesErrorOnInvalidParameter(self):
        with pytest.raises(TypeError):
            RendererBuilder().exportVideo(31)
        with pytest.raises(TypeError):
            RendererBuilder().exportVideo('/a/path/to/a/file', fps='not a number')

    def test_exportVideoAddsVideoExporter(self, basicRendererBuilder):
        renderer = basicRendererBuilder.exportVideo('test.mp4').build()
        assert next((observer for observer in renderer.observers
                     if isinstance(observer, RendererVideoExporter)), None) is not None

    def test_noVideoExportedByDefault(self, basicRendererBuilder):
        renderer = basicRendererBuilder.build()
        assert next((observer for observer in renderer.observers
                     if isinstance(observer, RendererVideoExporter)), None) is None
    # endregion

    # region withProgressBar
    def test_withProgressBarRaisesErrorOnInvalidParameter(self):
        with pytest.raises(TypeError):
            RendererBuilder().withProgressBar('not a bool')
        with pytest.raises(TypeError):
            RendererBuilder().withProgressBar(32)

    def test_withProgressBarAddsProgressBar(self, basicRendererBuilder):
        renderer = basicRendererBuilder.withProgressBar().build()
        assert next((observer for observer in renderer.observers
                     if isinstance(observer, RendererProgressBar)), None) is not None

    def test_noProgressBarByDefault(self, basicRendererBuilder):
        renderer = basicRendererBuilder.build()
        assert next((observer for observer in renderer.observers
                     if isinstance(observer, RendererProgressBar)), None) is None
    # endregion

    # region withLivePreview
    def test_withLivePreviewRaisesErrorOnInvalidParameter(self):
        with pytest.raises(TypeError):
            RendererBuilder().withLivePreview('not a bool')
        with pytest.raises(TypeError):
            RendererBuilder().withLivePreview(True, 'not a number')

    def test_withLivePreviewAddsLivePreview(self, basicRendererBuilder):
        renderer = basicRendererBuilder.withLivePreview().build()
        assert next((observer for observer in renderer.observers
                     if isinstance(observer, RendererLivePreview)), None) is not None

    def test_noLivePreviewByDefault(self, basicRendererBuilder):
        renderer = basicRendererBuilder.build()
        assert next((observer for observer in renderer.observers
                     if isinstance(observer, RendererLivePreview)), None) is None
    # endregion

    # region build
    def test_buildRaisesErrorIfNotAllRequiredAttributesSpecified(self, imageBatch, model, optimizer, objective, trainTFMS, drawTFMS):
        with pytest.raises(AttributeError):
            (RendererBuilder()
             .model(model)
             .optimizer(optimizer)
             .objective(objective)
             .trainTFMS(trainTFMS)
             .drawTFMS(drawTFMS)
             .build())
        with pytest.raises(AttributeError):
            (RendererBuilder()
             .imageBatch(imageBatch)
             .optimizer(optimizer)
             .objective(objective)
             .trainTFMS(trainTFMS)
             .drawTFMS(drawTFMS)
             .build())
        with pytest.raises(AttributeError):
            (RendererBuilder()
             .imageBatch(imageBatch)
             .model(model)
             .objective(objective)
             .trainTFMS(trainTFMS)
             .drawTFMS(drawTFMS)
             .build())
        with pytest.raises(AttributeError):
            (RendererBuilder()
             .imageBatch(imageBatch)
             .model(model)
             .optimizer(optimizer)
             .trainTFMS(trainTFMS)
             .drawTFMS(drawTFMS)
             .build())
        with pytest.raises(AttributeError):
            (RendererBuilder()
             .imageBatch(imageBatch)
             .model(model)
             .optimizer(optimizer)
             .objective(objective)
             .drawTFMS(drawTFMS)
             .build())
        with pytest.raises(AttributeError):
            (RendererBuilder()
             .imageBatch(imageBatch)
             .model(model)
             .optimizer(optimizer)
             .objective(objective)
             .trainTFMS(trainTFMS)
             .build())

    def test_buildConstructsRendererWithCorrectObjects(self, imageBatch, model, optimizer, objective, trainTFMS, drawTFMS):
        renderer = (RendererBuilder()
                    .imageBatch(imageBatch)
                    .model(model)
                    .optimizer(optimizer)
                    .objective(objective)
                    .trainTFMS(trainTFMS)
                    .drawTFMS(drawTFMS)
                    .build())
        assert renderer.imageBatch is imageBatch
        assert renderer.model is model
        assert renderer.optimizer is optimizer
        assert renderer.objective is objective
        assert renderer.trainTFMS is trainTFMS
        assert renderer.drawTFMS is drawTFMS
    # endregion
