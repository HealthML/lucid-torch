from Objective import Objective


class FCNeuronObjective(Objective):
    def __init__(self, get_layer, neuron=0):
        super().__init__(get_layer)
        self.neuron = neuron

    def _hook(self, module, input, output):
        self.output = output[:, self.neuron]
