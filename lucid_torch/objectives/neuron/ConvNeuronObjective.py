import numpy as np

from lucid_torch.objectives import Objective


class ConvNeuronObjective(Objective):
    def __init__(self, get_layer, channel=0, n_x=None, n_y=None):
        super().__init__(get_layer)
        self.channel = channel
        self.n_x, self.n_y = n_x, n_y

    def _hook(self, module, input, output):
        if self.n_x is None:
            n_y, n_x = np.array(output.shape[-2:]) // 2
        else:
            n_x, n_y = self.n_x, self.n_y
        self.output = output[:, self.channel, n_x, n_y]
