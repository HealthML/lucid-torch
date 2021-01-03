import numpy as np
import torch

from ..Objective import Objective


class ConvNeuronDiversityObjective(Objective):
    def __init__(self, get_layer, channel=0, n_x=None, n_y=None):
        super().__init__(get_layer)
        self.channel = channel
        self.n_x, self.n_y = n_x, n_y

    def _hook(self, module, input, output):
        if self.n_x is None:
            n_y, n_x = np.array(output.shape[-2:]) // 2
        else:
            n_x, n_y = self.n_x, self.n_y
        b = output.shape[0]
        flattened = output[:, self.channel, n_x, n_y].view(b, -1)
        gram = flattened @ flattened.t()
        gram = gram / gram.norm(p=2)
        gram = torch.triu(gram, diagonal=1)
        self.output = -gram.sum(1)
