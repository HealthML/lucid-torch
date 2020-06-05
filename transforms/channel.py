import torch
from typing import Union, List


class TFMSTransformChannels(torch.nn.Module):
    def __init__(self, channels: Union[int, List, torch.Tensor], tfms: torch.nn.Module):
        super(TFMSTransformChannels, self).__init__()
        if isinstance(channels, (list, int)):
            if isinstance(channels, int):
                if channels < 0:
                    raise ValueError()
                else:
                    channels = [channels]
            channels = torch.Tensor(channels)
        elif not isinstance(channels, torch.Tensor):
            raise TypeError()
        if not isinstance(tfms, torch.nn.Module):
            raise TypeError()
        self.channels = channels.long()
        self.tfms = tfms

    def forward(self, data: torch.Tensor):
        return data.index_copy(1, self.channels, self.tfms(data[:, self.channels]))
