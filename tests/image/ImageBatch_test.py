import numpy as np
import PIL
import pytest
import torch
import torchvision

from lucid_torch.image import ImageBatch

# region test image
EXAMPLE_IMAGE_PATH = 'tests/image/example_image.png'


@pytest.fixture
def example_image():
    pilImage = PIL.Image.open(EXAMPLE_IMAGE_PATH).convert('RGB')
    return torchvision.transforms.ToTensor()(pilImage)

# endregion

# region transform test doubles


class TFMSSetTo(torch.nn.Module):
    def __init__(self, data):
        super(TFMSSetTo, self).__init__()
        self.data = data

    def forward(self, data):
        return self.data
# endregion


class TestImageBatch:
    # region __init__
    def test_init_raises_error_on_invalid_parameters(self):
        with pytest.raises(TypeError):
            ImageBatch("not convertible to tensor")
        with pytest.raises(TypeError):
            class NotConvertibleToTensor:
                pass
            ImageBatch(NotConvertibleToTensor())

    def test_init_converts_parameters_to_tensor(self):
        def assert_converts(unconverted, converted: torch.Tensor):
            assert converted.float().equal(ImageBatch(unconverted).data.float())
        _float = 0.5
        _int = 2
        _list = [0.2, 0.3, 0.4]
        _ndarray = np.array([0.4, 0.5, 0.6])
        assert_converts(_float, torch.Tensor([_float]))
        assert_converts(_int, torch.Tensor([_int]))
        assert_converts(_list, torch.Tensor(_list))
        assert_converts(_ndarray, torch.from_numpy(_ndarray))
    # endregion

    # region load
    def test_load_raises_error_on_invalid_parameters(self):
        with pytest.raises(TypeError):
            ImageBatch.load(43)
        with pytest.raises(TypeError):
            ImageBatch.load(
                EXAMPLE_IMAGE_PATH, data_space_transform="not a transform")

    def test_load_loads_image(self, example_image):
        imageBatch = ImageBatch.load(EXAMPLE_IMAGE_PATH, data_space_transform=torch.nn.Identity())
        assert torch.all((example_image == imageBatch.data[0]))

    def test_load_applies_transforms(self):
        testData = torch.Tensor([-123.4, 567.8])
        transform = TFMSSetTo(testData)
        imageBatch = ImageBatch.load(
            EXAMPLE_IMAGE_PATH, data_space_transform=transform)
        assert testData.equal(imageBatch.data)
    # endregion

    # region generate
    def test_generate_raises_error_on_invalid_parameters(self):
        def assert_raises(width=32, height=32, batch_size=1, dimensions=3, std=0.2, ErrorT=TypeError):
            with pytest.raises(ErrorT):
                ImageBatch.generate(width=width, height=height, batch_size=batch_size,
                                    channels=dimensions, std=std)
        not_int_list = ["not a uint", 0.4, [1]]
        for not_uint in not_int_list:
            assert_raises(width=not_uint)
            assert_raises(height=not_uint)
            assert_raises(batch_size=not_uint)
            assert_raises(dimensions=not_uint)

        assert_raises(width=-10, ErrorT=ValueError)
        assert_raises(height=-10, ErrorT=ValueError)
        assert_raises(batch_size=-10, ErrorT=ValueError)
        assert_raises(dimensions=-10, ErrorT=ValueError)
        assert_raises(width=0, ErrorT=ValueError)
        assert_raises(height=0, ErrorT=ValueError)
        assert_raises(batch_size=0, ErrorT=ValueError)
        assert_raises(dimensions=0, ErrorT=ValueError)

        not_float_list = ["not a float", [0.2]]
        for not_float in not_float_list:
            assert_raises(std=not_float)

    def test_generate_applies_transforms(self):
        testData = torch.Tensor([-123.4, 567.8])
        transform = TFMSSetTo(testData)
        imageBatch = ImageBatch.generate(
            width=1, height=1, data_space_transform=transform)
        assert testData.equal(imageBatch.data)

    def test_generate_creates_correct_shape(self):
        assert ImageBatch.generate(
            width=24, height=33, batch_size=2, channels=1, data_space_transform=torch.nn.Identity()).data.shape == (2, 1, 33, 24)
        # endregion

    # region unmodified
    def test_links_to_unmodified(self):
        unmodified = ImageBatch(torch.Tensor([]))

        assert unmodified.unmodified() == unmodified

        modified = unmodified.transform(torch.nn.Identity())
        assert modified.unmodified() == unmodified
    # endregion

    # region transform
    def test_transform_raises_error_on_invalid_parameters(self):
        imageBatch = ImageBatch([0, 1, 2])
        with pytest.raises(TypeError):
            imageBatch.transform('not a transform')
        with pytest.raises(TypeError):
            imageBatch.transform([torch.nn.Identity])

    def test_transform_applies_transform(self):
        imageBatch = ImageBatch([0, 1, 2])
        data = torch.Tensor([3, 4, 5])
        assert data.equal(imageBatch.transform(TFMSSetTo(data)).data)

    def test_transform_creates_new_ImageBatch(self):
        original = ImageBatch([0, 1, 2])
        original_data = original.data.clone()
        assert original != original.transform(TFMSSetTo([3, 4, 5]))
        assert original.data.equal(original_data)
        # endregion

    # region dimensions
    def test_returns_correct_dimensions(self):
        w = 1
        h = 2
        d = 4
        b = 3
        imageBatch = ImageBatch.generate(
            width=w, height=h, batch_size=b, channels=d)
        assert imageBatch.width == imageBatch.data.shape[3]
        assert imageBatch.height == imageBatch.data.shape[2]
        assert imageBatch.channels == imageBatch.data.shape[1]
        assert imageBatch.batch_size == imageBatch.data.shape[0]

    # endregion
