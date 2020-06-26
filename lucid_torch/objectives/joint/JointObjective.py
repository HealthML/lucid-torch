from lucid_torch.objectives import Objective


class JointObjective(Objective):
    def __init__(self, objectives, weights=None):
        super().__init__(None)
        self.objectives = objectives
        if weights is None:
            self.weights = [1. for _ in self.objectives]
        else:
            self.weights = weights

    def register(self, model):
        for obj in self.objectives:
            obj.register(model)

    def remove_hook(self):
        for obj in self.objectives:
            obj.remove_hook()

    def _compute_loss(self, imageBatch):
        self.imageBatch = imageBatch
        loss = 0.
        for w, o in zip(self.weights, self.objectives):
            loss += w * o._compute_loss(self.imageBatch)
        return loss
