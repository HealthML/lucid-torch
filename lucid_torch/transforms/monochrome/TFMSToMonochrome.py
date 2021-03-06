from typing import List

import torch


class TFMSToMonochrome(torch.nn.Module):
    def __init__(self, factors: List[float] = [0.2126, 0.7152, 0.0722]):
        super(TFMSToMonochrome, self).__init__()
        if not isinstance(factors, list):
            raise TypeError()
        self.factors = torch.Tensor([factors]).unsqueeze(-1).unsqueeze(-1)

    def forward(self, data: torch.Tensor):
        if not data.shape[1] == self.factors.shape[1]:
            raise ValueError()
        self.factors = self.factors.to(data.device)
        data = data * self.factors
        data = data.sum(1, keepdim=True)
        return data
