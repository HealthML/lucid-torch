import torch

from lucid_torch.examples.alpha import alpha
from lucid_torch.examples.basic import basic
from lucid_torch.examples.objective import objective


def test_alphaReturnsTensor():
    assert isinstance(alpha('cpu', 1), torch.Tensor)


def test_basicReturnsTensor():
    assert isinstance(basic('cpu', 1), torch.Tensor)


def test_objectiveReturnsTensor():
    assert isinstance(objective('cpu', 1), torch.Tensor)
