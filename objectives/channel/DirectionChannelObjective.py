import torch

from objectives.Objective import Objective


class DirectionChannelObjective(Objective):
    def __init__(self, get_layer, direction):
        super().__init__(get_layer)
        self.direction = torch.tensor(
            direction, dtype=torch.float32).view(1, -1)

    def _hook(self, module, input, output):
        self.output = output[:, :, :, :]

    def _compute_loss(self, imageBatch):
        self.imageBatch = imageBatch
        self.direction = self.direction.to(self.output.device)
        n_neurons = self.output.shape[-1] * self.output.shape[-2]
        output = self.output.sum(-1).sum(-1) / n_neurons
        loss = -torch.cosine_similarity(output, self.direction)
        return loss.mean()
