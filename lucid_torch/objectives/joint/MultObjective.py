from .JointObjective import JointObjective


class MultObjective(JointObjective):
    def _compute_loss(self, imageBatch):
        self.imageBatch = imageBatch
        loss = 1.0
        for o in self.objectives:
            loss = o._compute_loss(self.imageBatch) * loss
        return loss
