from abc import ABC, abstractmethod

import numpy as np

import torch


class Objective(ABC):
    # TODO make sum([obj1, obj2, ...]) work
    def __init__(self, get_layer):
        self.get_layer = get_layer

    def register(self, model):
        self.layer_hook = self.get_layer(
            model).register_forward_hook(self._hook)

    def remove_hook(self):
        self.layer_hook.remove()

    def _compute_loss(self):
        loss = -1 * self.output.mean()
        return loss

    def backward(self):
        val = self._compute_loss()
        val.backward()

    def __add__(self, other):
        return JointObjective([self, other])

    def __neg__(self):
        return JointObjective([self], [-1.])

    def __sub__(self, other):
        return JointObjective([self, other], [1., -1.])

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return JointObjective([self], [other])
        else:
            raise NotImplementedError

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)

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

    def _compute_loss(self):
        loss = 0.
        for w, o in zip(self.weights, self.objectives):
            loss += w * o._compute_loss()
        return loss


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
        #self.output = output[:, self.channel, :, :]
        self.output = output[:, self.channel, :, :].sum([-1, -2])


class DirectionChannel(Objective):
    def __init__(self, get_layer, direction):
        super().__init__(get_layer)
        self.direction = torch.tensor(
            direction, dtype=torch.float32).view(1, -1)

    def _hook(self, module, input, output):
        self.output = output[:, :, :, :]

    def _compute_loss(self):
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
