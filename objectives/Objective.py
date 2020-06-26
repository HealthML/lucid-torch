from abc import ABC

from image.ImageBatch import ImageBatch


class Objective(ABC):
    def __init__(self, get_layer):
        self.get_layer = get_layer
        self.imageBatch = None

    def register(self, model):
        self.layer_hook = self.get_layer(
            model).register_forward_hook(self._hook)

    def remove_hook(self):
        self.layer_hook.remove()

    def _compute_loss(self, imageBatch: ImageBatch):
        self.imageBatch = imageBatch
        loss = -1 * self.output.mean()
        return loss

    def backward(self, imageBatch: ImageBatch):
        loss = self._compute_loss(imageBatch)
        loss.backward()
        return loss.item()

    def __add__(self, other):
        from objectives.joint.JointObjective import JointObjective
        from objectives.ConstObjective import ConstObjective
        if isinstance(other, (int, float)):
            other = ConstObjective(other)
        return JointObjective([self, other])

    def __neg__(self):
        from objectives.joint.JointObjective import JointObjective
        return JointObjective([self], [-1.])

    def __sub__(self, other):
        from objectives.joint.JointObjective import JointObjective
        from objectives.ConstObjective import ConstObjective
        if isinstance(other, (int, float)):
            other = ConstObjective(other)
        return JointObjective([self, other], [1., -1.])

    def __mul__(self, other):
        from objectives.joint.JointObjective import JointObjective
        from objectives.joint.MultObjective import MultObjective
        if isinstance(other, (int, float)):
            return JointObjective([self], [other])
        else:
            return MultObjective([self, other])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        from objectives.joint.JointObjective import JointObjective
        from objectives.ConstObjective import ConstObjective
        if isinstance(other, (int, float)):
            other = ConstObjective(other)
        return JointObjective([other, self], [1., -1.])

    def _hook(self, module, input, output):
        pass