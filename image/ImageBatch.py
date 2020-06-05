import warnings
from typing import Iterable, Union

import numpy as np
import PIL
import torch
from torchvision.transforms import ToTensor

warnings.filterwarnings('ignore', category=UserWarning)


class ImageBatch:
    @property
    def width(self):
        return self.data.shape[3]

    @property
    def height(self):
        return self.data.shape[2]

    @property
    def channels(self):
        return self.data.shape[1]

    @property
    def batch_size(self):
        return self.data.shape[0]

    def __init__(self, data: torch.Tensor):
        """
        data.shape = (batch, channels, height, width)
        """
        if isinstance(data, float):
            self.data = torch.tensor([data])
        elif isinstance(data, int):
            self.data = torch.tensor([data])
        elif isinstance(data, list):
            self.data = torch.tensor(data)
        elif isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data)
        elif isinstance(data, torch.Tensor):
            self.data = data
        else:
            raise TypeError()

    @staticmethod
    def load(paths: Union[str, Iterable[str]],
             data_space_transform: Union[torch.nn.Module, None] = None):
        if isinstance(paths, str):
            paths = [paths]
        elif not isinstance(paths, Iterable):
            raise TypeError()
        if data_space_transform is None:
            data_space_transform = torch.nn.Identity()
        elif not isinstance(data_space_transform, torch.nn.Module):
            raise TypeError()

        data = []
        for path in paths:
            img = PIL.Image.open(path).convert('RGB')
            data.append(ToTensor()(img).view(1, 3, img.height, img.width))
        data = torch.cat(data)
        return ImageBatch(data_space_transform(data))

    @staticmethod
    def generate(
            width: int = 224,
            height: int = 224,
            batch_size: int = 3,
            channels: int = 3,
            std: float = 0.01,
            data_space_transform: Union[torch.nn.Module, None] = None):
        def assertIsPositiveInt(value):
            if not isinstance(value, int):
                raise TypeError()
            elif value < 1:
                raise ValueError()

        def assertIsFloat(value):
            if not isinstance(value, (float, int)):
                raise TypeError()

        assertIsPositiveInt(width)
        assertIsPositiveInt(height)
        assertIsPositiveInt(batch_size)
        assertIsPositiveInt(channels)
        assertIsFloat(std)
        if data_space_transform is None:
            data_space_transform = torch.nn.Identity()
        elif not isinstance(data_space_transform, torch.nn.Module):
            raise TypeError()

        data = torch.normal(0.5,
                            std,
                            (batch_size, channels, height, width)).clamp(0.0, 1.0)
        return ImageBatch(data_space_transform(data))

    def unmodified(self):
        return self

    def transform(self, transforms: torch.nn.Module):
        if not isinstance(transforms, torch.nn.Module):
            raise TypeError()
        return ModifiedImageBatch(transforms(self.data), self)

    def to(self, device):
        self.data = torch.tensor(
            self.data, device=device, requires_grad=True, dtype=torch.float)
        return self


class ModifiedImageBatch(ImageBatch):
    def __init__(self,
                 data: torch.Tensor,
                 unmodified: ImageBatch):
        super(ModifiedImageBatch, self).__init__(data)
        self._unmodified = unmodified

    def unmodified(self):
        return self._unmodified
