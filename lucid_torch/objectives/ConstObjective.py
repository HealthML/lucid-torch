from ..image import ImageBatch
from .Objective import Objective


class ConstObjective(Objective):
    def __init__(self, const):
        super().__init__(None)
        if isinstance(const, (float, int)):
            self.const = const
        else:
            raise NotImplementedError

    def backward(self, imageBatch: ImageBatch):
        raise NotImplementedError

    def register(self, model):
        pass

    def remove_hook(self):
        pass

    def _compute_loss(self, imageBatch):
        self.imageBatch = imageBatch
        return self.const
