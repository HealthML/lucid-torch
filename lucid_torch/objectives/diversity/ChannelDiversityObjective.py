import torch

from ..Objective import Objective


class ChannelDiversityObjective(Objective):
    def __init__(self, get_layer, channel):
        super().__init__(get_layer)
        self.channel = channel

    def _hook(self, module, input, output):
        b = output.shape[0]
        flattened = output[:, self.channel].view(b, -1)
        gram = flattened @ flattened.t()
        gram = gram / gram.norm(p=2)
        gram = torch.triu(gram, diagonal=1)
        self.output = -gram.sum(1)
