import torch

from ...examples import alpha, basic, multiple_objectives


def test_alphaReturnsTensor():
    assert isinstance(alpha('cpu', 1), torch.Tensor)


def test_basicReturnsTensor():
    assert isinstance(basic('cpu', 1), torch.Tensor)


def test_objectiveReturnsTensor():
    assert isinstance(multiple_objectives('cpu', 1), torch.Tensor)
