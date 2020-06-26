import torch


class TFMSMonochromeTo(torch.nn.Module):
    def __init__(self, num_dimensions: int = 3):
        super(TFMSMonochromeTo, self).__init__()
        if not isinstance(num_dimensions, int):
            raise TypeError()
        elif num_dimensions < 2:
            raise ValueError()
        self.num_dimensions = num_dimensions

    def forward(self, data: torch.Tensor):
        return data.expand(data.shape[0], self.num_dimensions, *data.shape[2:])
