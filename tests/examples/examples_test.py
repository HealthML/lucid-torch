import torch

from lucid_torch.examples import alpha, basic, objective


def test_alphaReturnsTensor():
    assert isinstance(alpha('cpu', 1), torch.Tensor)


def test_basicReturnsTensor():
    assert isinstance(basic('cpu', 1), torch.Tensor)


def test_objectiveReturnsTensor():
    assert isinstance(objective('cpu', 1), torch.Tensor)
