from objectives.Objective import Objective


class ChannelObjective(Objective):
    # TODO: speed up when only one objective is given (i.e. no full forward pass)
    # TODO: several objectives (i.e. self.output no single but several
    # TODO: different aggregation functions besides mean/sum
    '''
    e.g. channel = 10
    get_layer = lambda model: model.layer1[0].conv1
    '''

    def __init__(self, get_layer, channel):
        super().__init__(get_layer)
        self.channel = channel

    def _hook(self, module, input, output):
        # TODO check which one is correct/better!
        # TODO then propagate for all objectives!!
        # self.output = output[:, self.channel, :, :]
        self.output = output[:, self.channel, :, :].sum([-1, -2])
