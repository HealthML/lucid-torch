import torch

from examples.alpha import alpha
from examples.basic import basic
from examples.objective import objective


def test_alphaReturnsTensor():
    assert isinstance(alpha('cpu', 1), torch.Tensor)


def test_basicReturnsTensor():
    assert isinstance(basic('cpu', 1), torch.Tensor)


def test_objectiveReturnsTensor():
    assert isinstance(objective('cpu', 1), torch.Tensor)
