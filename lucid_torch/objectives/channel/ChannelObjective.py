from ..Objective import Objective


class ChannelObjective(Objective):
    def __init__(self, get_layer, channel):
        super().__init__(get_layer)
        self.channel = channel

    def _hook(self, module, input, output):
        self.output = output[:, self.channel, :, :].sum([-1, -2])
