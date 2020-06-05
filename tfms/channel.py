from typing import List, Union

import torch


class TFMSTransformChannels(torch.nn.Module):
    def __init__(self, channels: Union[int, List, torch.Tensor], tfms: torch.nn.Module):
        super(TFMSTransformChannels, self).__init__()
        if isinstance(channels, (list, int)):
            if isinstance(channels, int):
                channels = [channels]
            channels = torch.Tensor(channels)
        elif not isinstance(channels, torch.Tensor):
            raise TypeError()
        if not isinstance(tfms, torch.nn.Module):
            raise TypeError()
        self.channels = channels.long()
        self.tfms = tfms

    def forward(self, data: torch.Tensor):
        self.channels = self.channels.to(data.device)
        transformed = self.tfms(data[:, self.channels])
        data = data.clone()
        data[:, self.channels] = transformed
        return data
