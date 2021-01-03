import torch

from ..Objective import Objective


class FCDiversityObjective(Objective):
    # TODO different gram aggregation (e.g. max)
    def __init__(self, get_layer, neuron=0):
        super().__init__(get_layer)
        self.neuron = neuron

    def _hook(self, module, input, output):
        b = output.shape[0]
        flattened = output[:, self.neuron].view(b, -1)
        gram = flattened @ flattened.t()
        gram = gram / gram.norm(p=2)
        gram = torch.triu(gram, diagonal=1)
        self.output = -gram.sum(1)
