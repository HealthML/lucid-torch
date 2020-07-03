from lucid_torch.objectives import Objective


class LayerObjective(Objective):
    '''
    get_layer = lambda model: model.layer1[0].conv1
    '''

    def __init__(self, get_layer):
        super().__init__(get_layer)

    def _hook(self, module, input, output):
        self.output = output[:, :, :, :]
