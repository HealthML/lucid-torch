from abc import ABC
from image.ImageBatch import ImageBatch

import numpy as np
import torch


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
        if isinstance(other, (int, float)):
            other = ConstObjective(other)
        return JointObjective([self, other])

    def __neg__(self):
        return JointObjective([self], [-1.])

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = ConstObjective(other)
        return JointObjective([self, other], [1., -1.])

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return JointObjective([self], [other])
        else:
            return MultObjective([self, other])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = ConstObjective(other)
        return JointObjective([other, self], [1., -1.])

    def _hook(self, module, input, output):
        pass


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


class MultObjective(JointObjective):
    def _compute_loss(self, imageBatch):
        self.imageBatch = imageBatch
        loss = 1.0
        for o in self.objectives:
            loss = o._compute_loss(self.imageBatch) * loss
        return loss


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


class ConvNeuron(Objective):
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


class FCNeuron(Objective):
    def __init__(self, get_layer, neuron=0):
        super().__init__(get_layer)
        self.neuron = neuron

    def _hook(self, module, input, output):
        self.output = output[:, self.neuron]


class Channel(Objective):
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


class DirectionChannel(Objective):
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


class ConvNeuronDiversity(Objective):
    # TODO different gram aggregation (e.g. max)
    def __init__(self, get_layer, channel=0, n_x=None, n_y=None):
        super().__init__(get_layer)
        self.channel = channel
        self.n_x, self.n_y = n_x, n_y

    def _hook(self, module, input, output):
        if self.n_x is None:
            n_y, n_x = np.array(output.shape[-2:]) // 2
        else:
            n_x, n_y = self.n_x, self.n_y
        b = output.shape[0]
        flattened = output[:, self.channel, n_x, n_y].view(b, -1)
        gram = flattened @ flattened.t()
        gram = gram / gram.norm(p=2)
        gram = torch.triu(gram, diagonal=1)
        self.output = -gram.sum(1)


def get_diversity_like(obj):
    if isinstance(obj, Channel):
        return ChannelDiversity(obj.get_layer, obj.channel)
    elif isinstance(obj, FCNeuron):
        return FCDiversity(obj.get_layer, obj.neuron)
    else:
        raise NotImplementedError


class ChannelDiversity(Objective):
    # TODO different gram aggregation (e.g. max)
    def __init__(self, get_layer, channel):
        super().__init__(get_layer)
        self.channel = channel

    def _hook(self, module, input, output):
        b = output.shape[0]
        flattened = output[:, self.channel].view(b, -1)
        gram = flattened @ flattened.t()
        gram = gram / gram.norm(p=2)
        gram = torch.triu(gram, diagonal=1)
        self.output = -gram.sum(1)


class FCDiversity(Objective):
    # TODO different gram aggregation (e.g. max)
    def __init__(self, get_layer, neuron=0):
        super().__init__(get_layer)
        self.neuron = neuron

    def _hook(self, module, input, output):
        b = output.shape[0]
        flattened = output[:, self.neuron].view(b, -1)
        gram = flattened @ flattened.t()
        gram = gram / gram.norm(p=2)
        gram = torch.triu(gram, diagonal=1)
        self.output = -gram.sum(1)


class Layer(Objective):
    '''
    get_layer = lambda model: model.layer1[0].conv1
    '''

    def __init__(self, get_layer):
        super().__init__(get_layer)

    def _hook(self, module, input, output):
        self.output = output[:, :, :, :]


class ImageObjective(Objective):
    def __init__(self):
        super().__init__(None)

    def register(self, model):
        pass

    def remove_hook(self):
        pass


class MeanOpacity(ImageObjective):
    def __init__(self, unit_space_tfms: torch.nn.Module):
        super().__init__()
        self.unit_space_tfms = unit_space_tfms

    def _compute_loss(self, imageBatch):
        self.imageBatch = imageBatch
        unmodified = self.imageBatch.unmodified()
        opacity = unmodified.transform(self.unit_space_tfms).data[-1]
        return opacity.sigmoid().mean()
