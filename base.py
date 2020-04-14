## TODO
# use transformation library such as kornia or nvidia-dali instead of own ones

## TODO
# try other learning policies; e.g. OneCycle, multiple cycles, or other LR Schedulers?
# maybe even different lrs for different pixels? e.g. in center different than outside?

## TODO
# try out different color models; especially HSL, or maybe even fix SL (e.g. from a real image) and then only change hue (different learning rates for H, S, and L)? -> walk cyclically through hue? check out different hue parametrizations, such as Munsell, NCS

## TODO
# how will training the networks already in HSL (or other color space) change feature visualizations?

## TODO
# include proper transformations for visualizations --> they should match what we trained with, not weird resizes
# -> check everywhere, e.g. in initializer as well

## TODO
# find image patches that most highly activate channels instead of optimize! --> everywhere instead of fv?
# 

## TODO
# make feature importances more intuitive, e.g. via bars

## TODO
# dataset specific decorrelation!

## TODO resume alpha:
### -> clean everything up, currently via list and everything's ugly
### -> background image should be way smoother
### -> gamma correction
### -> regularize alpha *(1-mean(alpha))
### -> check lucid implementation again, they might have other changes

## TODO:
### what happens if we add zeroth/first layer + later layer objective? shouldn't this make sharp but meaningful images?
### ---> doesn't seem to work well! doesn't make anything more sharp


# TODO
# TODO change class structure: objectives into FeatureObjectives(get_layer) & ImageObjectives(**)
# --> transparent images (i.e. incl alpha channel
# --> several images at the same time if init from image
# --> diversity for full batch # TODO --> check whether this is correct -> looks different than in tf and doesn't work so well?


# TODO
# potentially: do border radius already here, due to performance issues!
# eg: integrate into build_spritemap
# prob doesn't really make sense -> then still have rectangle borders

# TODO
# better/other alternatives to CAM
# those right now are shitty and also sometimes constant...

# TODO:
# also show/return objective value for each generated image
# TODO:
# ImageObjective that penalize init from image images to look more like the original image (eg in ssim)

# TODO:
# create many new images via feature visualization and then --> can we use them for statistics?

# TODO
# combine optimization with CAM: focus feature visualization on those parts of an image that the cam shows to be most important for the clf

# TODO also show association between each neuron and genotype --> gives even more interpretability


# TODO git!!!!

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from abc import ABC, abstractmethod
from functools import partial
import os
from os.path import join
import json
import pickle
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from skimage.io import imread, imsave
from skimage.transform import resize as sk_resize

from PIL import Image

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import models
from torchvision.transforms import ToTensor, Resize

def img_from_param(size=(1, 3, 128, 128), std=0.01, fft=True, decorrelate=True, decay_power=1, seed=42, dev='cpu', path=None, eps=1e-5):
    if not path is None:
        img, pre, post = init_from_image(
                path,
                size=size[-2:],
                fft=fft,
                dev=dev,
                eps=eps
                ) 
    else:
        img, pre, post = get_image(
                size=size,
                std=std,
                fft=fft,
                decay_power=decay_power,
                seed=seed,
                dev=dev,
                )
    to_rgb = partial(to_valid_rgb,
            pre_correlation=pre, post_correlation=post, decorrelate=decorrelate)
    return img, to_rgb

def opt_from_param(img, opt='adam', lr=0.05, eps=1e-7, wd=0.):
    if opt == 'adam':
        opt = optim.Adam([img], lr=lr, eps=1e-7, weight_decay=wd)
    else:
        raise NotImplementedError
    return opt

def tfms_from_param(tfm_param='default'):
    if isinstance(tfm_param, str) and tfm_param == 'default':
        tfm_param = [
            TFMSPad(12, 'constant', 0.5),
            TFMSJitter(8),
            TFMSRandomScale([1 + (i - 5) / 50. for i in range(11)]),
            TFMSRandomRotate(list(range(-10, 11)) + 5 * [0]),
            TFMSJitter(4),
            ]
    elif isinstance(tfm_param, str) and tfm_param == 'default_norm':
        tfm_param = [
            TFMSPad(12, 'constant', 0.5),
            TFMSJitter(8),
            TFMSRandomScale([1 + (i - 5) / 50. for i in range(11)]),
            TFMSRandomRotate(list(range(-10, 11)) + 5 * [0]),
            TFMSJitter(4),
            TFMSNormalize(),
            ]

    elif not isinstance(tfm_param, list):
        raise NotImplementedError

    return TFMSCompose(tfm_param)
  
def render(model, objective, img_thres=(100,),
        img_param={}, opt_param={}, tfm_param='default',
        seed=None, dev='cuda:0',
        verbose=True,
        ):
    if seed:
        torch.manual_seed(seed)

    model = prep_model(model, dev)
    objective.register(model)

    img, to_rgb = img_from_param(dev=dev, **img_param)
    opt = opt_from_param(img, **opt_param)
    tfms = tfms_from_param(tfm_param)

    imgs = []
    if verbose:
        pbar = tqdm(range(1, max(img_thres)+1))
    else:
        pbar = range(1, max(img_thres)+1)
    for i in pbar:
        step(img, opt, objective, model, to_rgb, tfms)
        if verbose:
            with torch.no_grad():
                pbar.set_description("Epoch %d, current loss: %.3f" % (i, objective._compute_loss()))
        if i in img_thres:
            imgs.append(to_rgb(img).detach().cpu().numpy())

    imgs = np.moveaxis(np.array(imgs), 2, -1)
    if verbose:
        plot_imgs(imgs[-1])

    objective.remove_hook()
    return imgs


def plot_imgs(imgs):
    '''plot imgs into joint figure
    
    imgs: (batch, height, width, channels)
    '''
    n_img = imgs.shape[0]
    if n_img < 17:
        n_rows = [None, 1, 1, 2, 2, 2, 2, 3, 3, 3,  3,  3,  3,  4,  4,  4,  4][n_img]
    else:
        n_rows = 4
    n_cols = np.ceil(n_img / n_rows).astype(int)
    fig = plt.figure(figsize=(10, 10))
    for i, img in enumerate(imgs):
        fig.add_subplot(n_rows,  n_cols, i+1)
        plt.imshow(img)


def step(img, opt, obj, model, to_rgb, tfms):
    opt.zero_grad()
    # e.g. sigmoid, or inverse fft
    img = to_rgb(img)
    if not tfms is None:
        img = tfms(img)
    _ = model(img)
    obj.backward()
    opt.step()

#############################################################
######## START OBJECTIVES ###################################
#############################################################


class Objective(ABC):
    # TODO make sum([obj1, obj2, ...]) work
    def __init__(self, get_layer):
        self.get_layer = get_layer

    def register(self, model):
        self.layer_hook = self.get_layer(model).register_forward_hook(self._hook)

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
        self.output = output[:, self.channel, :, :].sum([-1,-2])



class DirectionChannel(Objective):
    def __init__(self, get_layer, direction):
        super().__init__(get_layer)
        self.direction = torch.tensor(direction, dtype=torch.float32).view(1, -1)

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


#############################################################
######## END OBJECTIVES #####################################
#############################################################

#############################################################
######## START TRANSFORMS ###################################
#############################################################

class TFMS(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, img):
        pass

class TFMSNormalize(TFMS):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    def __call__(self, img):
        dev = img.device
        self.mean, self.std = self.mean.to(dev), self.std.to(dev)
        return (img - self.mean[..., None, None]) / self.std[..., None, None]

class TFMSRandomScale(TFMS):
    def __init__(self, scales=None, rng=None, mode='bilinear'):
        self.scales = scales
        self.rng = rng
        if (scales is None) and (rng is None):
            raise ValueError
        self.mode = mode

    def __call__(self, img):
        if not self.scales is None:
            scale = np.random.choice(self.scales)
        else:
            scale = self.rng[0] + (self.rng[1] - self.rng[0]) * np.random.rand()
        return F.interpolate(img, scale_factor=scale, mode=self.mode, align_corners=False)



class TFMSJitter(TFMS):
    '''jitter in the lucid sense, not in the standard definition'''
    def __init__(self, d):
        self.d = d
    def __call__(self, img):
        w, h = img.shape[-2:]
        w_start, h_start = np.random.choice(self.d, 2)
        w_end, h_end = w + w_start - self.d, h + h_start - self.d
        return img[:,:, w_start:w_end, h_start:h_end]

class TFMSPad(TFMS):
    def __init__(self, w, mode='constant', constant_value=0.5):
        self.w = w
        self.mode = mode
        self.constant_value = constant_value

    def __call__(self, img):
        if self.constant_value == 'uniform':
            val = torch.rand(1)
        else:
            val = self.constant_value
        pad = nn.ConstantPad2d(self.w, val)
        return pad(img)

class TFMSRandomRotate(TFMS):
    def __init__(self, angles=None, rng=None):
        self.angles = angles
        if not angles is None:
            self.rotations = []
            for a in angles:
                self.rotations.append(TFMSRotateGPU(a))
        self.rng = rng
        if (angles is None) and (rng is None):
            raise ValueError
    def __call__(self, img):
        if not self.angles is None:
            rot = np.random.choice(self.rotations)
        else:
            angle = self.rng[0] + (self.rng[1] - self.rng[0]) * np.random.rand()
            rot = TFMSRotateGPU(angle)

        return rot(img)

class TFMSRotateGPU(TFMS):
    def __init__(self, angle=0):
        self.angle = torch.tensor([angle*np.pi / 180])
        self.rot_matrix = torch.tensor([[torch.cos(self.angle), torch.sin(self.angle)],
                                    [-torch.sin(self.angle), torch.cos(self.angle)]])
        self.w = self.h = None

    def _set_up(self, w, h, dev):
        self.rot_matrix = self.rot_matrix.to(dev)
        self.w, self.h = w, h
        xx, yy = torch.meshgrid(torch.arange(w, device=dev), torch.arange(h, device=dev))
        xx, yy = xx.contiguous().float(), yy.contiguous().float()
        xm, ym = (w + 1) / 2, (h + 1) / 2

        inds = torch.cat([(xx - xm).view(-1, 1), (yy - ym).view(-1, 1)], dim=1)
        inds = (self.rot_matrix @ inds.t()).round() + torch.tensor([[xm, ym]], device=dev).t()

        inds[inds < 0] = 0.
        inds[0, :][inds[0, :] >= w] = w - 1.
        inds[1, :][inds[1, :] >= h] = h - 1.
        self.inds = inds.long()
        self.xx, self.yy = xx.long(), yy.long()

    def __call__(self, img):
        w, h = img.shape[-2:]
        dev = img.device
        if not (self.w, self.h) == (w, h):
            self._set_up(w, h, dev)

        rot_img = torch.zeros_like(img)
        rot_img[:, :, self.xx.view(-1), self.yy.view(-1)] = img[:, :, self.inds[0, :], self.inds[1, :]]

        return rot_img

class TFMSGaussianNoise(TFMS):
    def __init__(self, level=0.01):
        self.level = level

    def __call__(self, img):
        return img + self.level * torch.randn_like(img)

class TFMSBlur(TFMS):
    def __init__(self, kernel=torch.ones(3, 3)):
        self.kernel = kernel.view(1, 1, *kernel.shape).repeat(3, 1, 1, 1)
        self.pad = kernel.shape[0] // 2

    def __call__(self, img):
        self.kernel = self.kernel.to(img.device)
        #return F.conv2d(img.view(1, *img.shape), weight=self.kernel, groups=3, padding=self.pad)[0]
        return F.conv2d(img, weight=self.kernel, groups=3, padding=self.pad)

class TFMSGaussianBlur(TFMSBlur):
    def __init__(self, kernel_size=3, std=1):
        ax = torch.linspace(-(kernel_size - 1)/2., (kernel_size - 1) /2, kernel_size)
        xx, yy = torch.meshgrid(ax, ax)
        kernel = torch.exp(-0.5 * (xx**2 + yy**2) / std**2)
        super(TFMSGaussianBlur, self).__init__(kernel=kernel/kernel.sum())



    



class TFMSRotate(TFMS):
    def __init__(self, angle=0):
        self.angle = torch.tensor([angle*np.pi / 180])
        self.rot_matrix = torch.tensor([[torch.cos(self.angle), torch.sin(self.angle)],
                                    [-torch.sin(self.angle), torch.cos(self.angle)]])

    def __call__(self, img):
        w, h = img.shape[-2:]

        xx, yy = torch.meshgrid(torch.arange(w), torch.arange(h))
        xx, yy = xx.contiguous().float(), yy.contiguous().float()
        xm, ym = (w + 1) / 2, (h + 1) / 2

        inds = torch.cat([(xx - xm).view(-1, 1), (yy - ym).view(-1, 1)], dim=1)
        inds = (self.rot_matrix @ inds.t()).round() + torch.tensor([[xm, ym]]).t()

        inds[inds < 0] = 0.
        inds[0, :][inds[0, :] >= w] = w - 1.
        inds[1, :][inds[1, :] >= h] = h - 1.
        inds = inds.long()

        rot_img = torch.zeros_like(img)
        rot_img[:, :, xx.view(-1).long(), yy.view(-1).long()] = img[:, :, inds[0, :], inds[1, :]]

        return rot_img

class TFMSCompose(TFMS):
    def __init__(self, tfms):
        self.tfms = tfms

    def __call__(self, img):
        for tfm in self.tfms:
            img = tfm(img)
        return img

############################################################
######## END TRANSFORMS #####################################
#############################################################




#############################################################
######## START IMAGE PARAMETRIZATION ########################
#############################################################


def init_from_image(paths, size=(224, 224), fft=False, dev='cuda:0', eps=1e-4):
    # TODO: 
    # -> fft
    # -> decorrelation?
    if not isinstance(paths, list):
        paths = [paths]
    imgs = []
    for path in paths:
        img = Image.open(path).convert('RGB').resize(size)
        img = ToTensor()(img).view(1, 3, *size)
        img[img == 1] = 1. - eps
        img[img == 0] = eps

        img = torch.log( img / (1. - img) ).to(dev)

        if fft:
            w, h = size
            freqs = rfft2d_freqs(*size)
            decay_power = 1
            scale = (torch.tensor(np.sqrt(w*h) / np.maximum(freqs, 1./max(w, h)) ** decay_power))
            scale = scale.view(1, 1, *scale.shape, 1).float().to(dev)

            img = torch.rfft(img, signal_ndim=2) / scale

        imgs.append(img)
    imgs = torch.cat(imgs)
    imgs.requires_grad_()
     
    if fft:
        def pre_correlation(img):
            img = scale * img
            rgb_img = torch.irfft(img, signal_ndim=2)
            rgb_img = rgb_img[:, :3, :size[-2], :size[-1]]
            return rgb_img
        post_correlation = lambda x: torch.sigmoid(x)
    else:
        pre_correlation = lambda x: x
        post_correlation = lambda x: torch.sigmoid(x)

    return imgs, pre_correlation, post_correlation


def get_image(size, std, fft=False, dev='cuda:0', seed=124, decay_power=1.):
    if fft:
        b, ch, h, w = size
        freqs = rfft2d_freqs(h, w)
        freq_size = (b, ch, *freqs.shape, 2)
        img = torch.normal(0, std, freq_size).to(dev).requires_grad_()

        scale = (torch.tensor(np.sqrt(w*h) / np.maximum(freqs, 1./max(w, h)) ** decay_power))
        scale = scale.view(1, 1, *scale.shape, 1).float().to(dev)
        def pre_correlation(img):
            scaled_img = scale * img
            rgb_img = torch.irfft(scaled_img, signal_ndim=2)
            rgb_img = rgb_img[:b, :ch, :h, :w]
            # TODO what? why? maybe approx the same scale as for pixel images?
            rgb_img.div_(4)
            return rgb_img

        def post_correlation(img):
            return torch.sigmoid(img)
        
    else:
        img = torch.normal(0, std, size).to(dev).requires_grad_()
        pre_correlation = lambda x: x
        def post_correlation(img):
            return torch.sigmoid(img)
    return img, pre_correlation, post_correlation

def rfft2d_freqs(h, w):
    '''compute 2D spectrum frequencies (from lucid)'''
    fy = np.fft.fftfreq(h)[:, None]
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[:((w // 2) + 2)]
    else:
        fx = np.fft.fftfreq(w)[:((w // 2) + 1)]
    return np.sqrt(fx*fx + fy*fy)


# from lucid (on imagenet?)
color_correlation_svd_sqrt = torch.tensor([[0.26, 0.09, 0.02],
                                         [0.27, 0.00, -0.05],
                                         [0.27, -0.09, 0.03]])

# from ukb retina data (but is it correct?)
color_correlation_svd_sqrt3 = torch.tensor([[ 0.179, -0.041, -0.005],
                                            [ 0.100,  0.049,  0.016],
                                            [ 0.042,  0.057, -0.017]])

# TODO check - is this how color_correlation_svd_sqrt is meant to be?
# data = ... # (n_samples, 3, w, h)
# B = data.mean([-1,-2])
# cov = np.cov(B.numpy().T)
# u, s, vt = np.linalg.svd(cov)
# color_correlation = u @ np.diag(s**0.5)
C = color_correlation_svd_sqrt
max_norm_svd_sqrt = torch.norm(C, dim=0).max()
cc_norm = C / max_norm_svd_sqrt


def to_valid_rgb(img, pre_correlation, post_correlation, decorrelate=False):
    img = pre_correlation(img)

    if decorrelate:
        img = linear_decorrelate(img)
    img = post_correlation(img)

    return img

def linear_decorrelate(img):
    '''multiply input by sqrt of empirical ImageNet color correlation matrix

    TODO: use your own color correlation matrix!
    '''
    img = img.permute(0, 2, 3, 1)
    img = (img.reshape(-1, 3) @ cc_norm.t().to(img.device)).view(*img.shape)
    img = img.permute(0, 3, 1, 2)
    return img


#############################################################
######## END IMAGE PARAMETRIZATION ##########################
#############################################################

#############################################################
######## START UTILS ########################################
#############################################################

def prep_model(model, dev):
    # TODO rename
    '''eval, move to dev & make ReLUs not inplace'''
    model = model.eval().to(dev)
    for mod in model.modules():
        if hasattr(mod, 'inplace'):
            mod.inplace = False
    return model


def compute_layer(inp, model, layer_func, dev, include_grad=False, targets=None):
    # TODO rename! maybe compute_layer_activations?
    '''

    # grad only works for models with scalar output!!!

    m = ...
    inp = torch.rand(1, 3, 224, 224)
    layer_func = lambda m: m.layer3[1].conv2
    dev = 'cuda:0'
    if True:
        targets = torch.ones((1,1))
        act, grad = compute_layer(inp, m, layer_func, dev, True, targets)
    else:
        act = compute_layer(inp, m, layer_func, dev, False)

    '''
    model = prep_model(model, dev)
    output = []
    hook = layer_func(model).register_forward_hook(lambda m, i, o: output.append(o.detach()))
    if include_grad:
        hook_backward = layer_func(model).register_backward_hook(lambda m, i, o: output.append(o))

    out = model(inp.to(dev))

    hook.remove()
    if include_grad:
        loss = nn.BCELoss()(torch.sigmoid(out), targets.to(dev))
        loss.backward()
        hook_backward.remove()

    return output

# been used
def build_spritemap(channel_imgs, border=0, int_format=False):
    # TODO rename to create_spritemap?
    # TODO 'border' to 'padding' or 'pad'?

    # channel_imgs eg (256, 224, 224, 3)
    if len(channel_imgs.shape) == 4:
        ch, h, w, _ = channel_imgs.shape
    else:
        ch, h, w = channel_imgs.shape

    n_rows = n_cols = np.ceil(ch**0.5).astype(int)
    # TODO remove this
    '''
    if int(ch**0.5)**2 == ch:
        n_rows = n_cols = int(ch**0.5)
    else:
        n_cols = np.ceil(ch**0.5).astype(int)
        n_rows = np.ceil(ch**0.5).astype(int)
    '''

    if len(channel_imgs.shape) == 4:
        sprite_size = ((n_rows-1)*(h+border) + h, (n_cols-1)*(w+border) + w, 3)
    else:
        sprite_size = ((n_rows-1)*(h+border) + h, (n_cols-1)*(w+border) + w)
    sprite = np.ones(sprite_size)
    i = j = 0
    for img in channel_imgs:
        hh, hp = i*(h+border), i*(h+border) + h
        ww, wp = j*(w+border), j*(w+border) + w
        sprite[hh:hp, ww:wp] = img
        j = (j+1) % n_cols
        if j == 0:
            i += 1
    if int_format:
        sprite = np.round(sprite*255).astype(np.uint8)
    return sprite


def namify(string):
    return '_'.join(string.split())

#############################################################
######## END UTILS ##########################################
#############################################################

#############################################################
######## START EXPORT #######################################
#############################################################


def export_full(
        model, layer_funcs, layer_names, image_paths,
        sdir='dump/',
        n_epochs=2500,
        size=(448, 448), 
        img_extension='jpg',
        n_images_vis_1=4, n_images_vis_2=16, n_images_vis_4=5, sem_dict_len=3,
        dev='cuda:0'):
    # TODO rename
    # TODO export meta data

    general_s = 'general'
    vis_1 = 'vis_1_data'
    vis_2 = 'vis_2_data'
    vis_4 = 'vis_4_data'
    
    n_images = max(n_images_vis_1, n_images_vis_2, n_images_vis_4)
    selected_paths = list(np.random.choice(image_paths, size=n_images, replace=False))

    # basic prep
    if not os.path.isdir(sdir):
        os.mkdir(sdir)
    if not os.path.isdir(join(sdir, general_s)):
        os.mkdir(join(sdir, general_s))
    if not isinstance(layer_funcs, list):
        layer_funcs = [layer_funcs]
        layer_names = [layer_names]

    # general exports
    print('creating feature visualization spritemaps...')
    for layer_func, layer_name in zip(layer_funcs, layer_names):
        print('\tstarting layer %s...' % layer_name)
        layer_fvs = prepare_layer(model, layer_func, n_epochs=n_epochs, size=(1, 3, *size), dev=dev)
        layer_spritemap = build_spritemap(layer_fvs, int_format=True)
        s = '%s.%s' % (namify(layer_name), img_extension)
        layer_s = join(sdir, general_s, s)
        imsave(layer_s, layer_spritemap)

        # TODO 
        # spritemap_urls = [...]

    # export for vis 1
    print('creating visualization 1...')
    dir_name = join(sdir, vis_1)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    for layer_func, layer_name in zip(layer_funcs, layer_names):
        print('\tstarting layer %s...' % layer_name)
        layer_dir = join(sdir, vis_1, namify(layer_name))
        imgs = save_v1(
                selected_paths[:n_images_vis_1],
                model,
                layer_func,
                sdir=layer_dir,
                size=size,
                dev=dev)

    print('\tsaving images...')
    path_to_imgs = [join(dir_name, 'sample_img%d.jpg'%d) for d in range(len(imgs))]
    for p, img in zip(path_to_imgs, imgs):
        img.save(p)

 
    # export for vis 2
    # TODO also include activations, especially for data images
    print('creating visualization 2...')
    dir_name = join(sdir, vis_2)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    save_v2(
            model,
            n_epochs,
            paths=selected_paths[:n_images_vis_2],
            all_paths=image_paths,
            sdir=dir_name,
            size=size,
            dev=dev
            )

    # export for vis 4
    # TODO maybe use most highly activated images instead of random ones?
    print('creating visualization 4...')
    dir_name = join(sdir, vis_4)
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    for layer_func, layer_name in zip(layer_funcs, layer_names):
        print('\tstarting layer %s...' % layer_name)
        layer_dir = join(sdir, vis_4, namify(layer_name))
        if not os.path.isdir(layer_dir):
            os.mkdir(layer_dir)
        path_to_json = join(layer_dir, 'activations_img%d.json')
        save_v4(
                selected_paths[:n_images_vis_4],
                model,
                layer_func,
                path_to_json=path_to_json,
                first=sem_dict_len,
                size=size,
                dev=dev
                )
    print('\tsaving images...')
    path_to_imgs = [join(dir_name, 'sample_img%d.jpg'%d) for d in range(n_images_vis_4)]
    for out_p, in_p in zip(path_to_imgs, selected_paths[:n_images_vis_4]):
        # TODO better resize
        img = Image.open(in_p).convert('RGB').resize(size)
        img.save(out_p)
 
def save_v4(paths, model, layer_func, path_to_json, first=3, sdir='.', size=(448, 448), dev='cuda:0'):
    for i, p in enumerate(paths):
        img = load_img(p, size, dev)
        sem_dict = prep_semantic_dicts(img, model, layer_func, first=first, dev=dev)
        with open(path_to_json % i, 'w') as f:
            json.dump(sem_dict, f)






 

def prep_v2(model, n_epochs, paths, all_paths, size=(448, 448), dev='cuda:0'):
    # TODO rename
    final_layer_func = lambda m: m
    obj = FCNeuron(final_layer_func, neuron=0)
    n_images = len(paths)
    
    # basic fv
    pos_act = render(model, obj, img_thres=(n_epochs,),
                img_param={'size':(n_images, 3, *size)},
                dev=dev,
                verbose=False,
                )
    neg_act = render(model, -obj, img_thres=(n_epochs,),
                img_param={'size':(n_images, 3, *size)},
                dev=dev,
                verbose=False,
                )

    # fv from image init
    img_param_init = {'size':(n_images, 3, *size), 'fft':False, 'decorrelate':False, 'path':paths}
    pos_act_init = render(model, obj, img_thres=(n_epochs,),
                img_param=img_param_init,
                dev=dev,
                verbose=False,
                )
    neg_act_init = render(model, -obj, img_thres=(n_epochs,),
                img_param=img_param_init,
                dev=dev,
                verbose=False,
                )

    # highest activation images
    activations = []
    for path in tqdm(all_paths):
        # TODO implement load_img
        img = load_img(path, size, dev)
        with torch.no_grad():
            act = model(img).item()
        activations.append(act)
    asort = np.argsort(activations)
    hi_act_ind = asort[-n_images:]
    lo_act_ind = asort[:n_images]
    # TODO 
    pos_act_imgs = np.array([np.array(Image.open(all_paths[i]).resize(size)) for i in hi_act_ind]) / 255.
    neg_act_imgs = np.array([np.array(Image.open(all_paths[i]).resize(size)) for i in lo_act_ind]) / 255.

    pos_act, neg_act = build_spritemap(pos_act[0], int_format=True), build_spritemap(neg_act[0], int_format=True)
    pos_act_init, neg_act_init = build_spritemap(pos_act_init[0], int_format=True), build_spritemap(neg_act_init[0], int_format=True)
    pos_act_imgs, neg_act_imgs = build_spritemap(pos_act_imgs, int_format=True), build_spritemap(neg_act_imgs, int_format=True)
    return pos_act, neg_act, pos_act_init, neg_act_init, pos_act_imgs, neg_act_imgs

    
def load_img(path, size, dev):
    # TODO also use this everywhere, eg init_from_image
    # TODO include better transforms, e.g. center crop
    img = Image.open(path).convert('RGB').resize(size)
    img = ToTensor()(img).view(1, 3, *size).to(dev)
    return img




# been used
def save_v1(paths, model, layer_func, sdir='.', size=(448, 448), dev='cuda:0'):
    As, imgs, sprites, attrs = prep_v1(paths, model, layer_func, size, dev)

    if not os.path.isdir(sdir):
        os.mkdir(sdir)

    path_to_avg_feats = join(sdir, 'avg_feature_importances.json')
    with open(path_to_avg_feats, 'w') as f:
        json.dump(As, f)

    path_to_sprites = [join(sdir, 'cam_sprite%d.jpg'%d) for d in range(len(sprites))]
    for p, sprite in zip(path_to_sprites, sprites):
        imsave(p, sprite)

    path_to_attrs = join(sdir, 'feature_importances.json')
    with open(path_to_attrs, 'w') as f:
        json.dump(attrs, f)

    return imgs

def save_v2(model, n_epochs, paths, all_paths, sdir='.', size=(448, 448), dev='cuda:0'):
    pos_act, neg_act, pos_act_init, neg_act_init, pos_act_imgs, neg_act_imgs = prep_v2(
                        model, n_epochs, paths, all_paths, size=size, dev=dev)
    img_extension = 'jpg'

    if not os.path.isdir(sdir):
        os.mkdir(sdir)

    path_to_pos_act = join(sdir, 'pos_act.%s' % img_extension)
    with open(path_to_pos_act, 'w') as f:
        imsave(path_to_pos_act, pos_act)

    path_to_neg_act = join(sdir, 'neg_act.%s' % img_extension)
    with open(path_to_neg_act, 'w') as f:
        imsave(path_to_neg_act, neg_act)

    path_to_pos_act_init = join(sdir, 'pos_act_init.%s' % img_extension)
    with open(path_to_pos_act_init, 'w') as f:
        imsave(path_to_pos_act_init, pos_act_init)

    path_to_neg_act_init = join(sdir, 'neg_act_init.%s' % img_extension)
    with open(path_to_neg_act_init, 'w') as f:
        imsave(path_to_neg_act_init, neg_act_init)

    path_to_pos_act_img = join(sdir, 'pos_act_img.%s' % img_extension)
    with open(path_to_pos_act_img, 'w') as f:
        imsave(path_to_pos_act_img, pos_act_imgs)

    path_to_neg_act_img = join(sdir, 'neg_act_img.%s' % img_extension)
    with open(path_to_neg_act_img, 'w') as f:
        imsave(path_to_neg_act_img, neg_act_imgs)

 










# been used in vis
def prepare_layer(model, layer_func, n_epochs=250, size=(1, 3, 224, 224), dev='cuda:0'):
    # TODO rename
    '''compute feature visualizations for each channel in layer'''
    b, ch, h, w = compute_layer(torch.normal(0, 0.01, size), model, layer_func, dev)[0].shape

    channel_imgs = []
    for c in tqdm(range(ch)):
        obj = Channel(layer_func, channel=c)
        channel_img = render(
                model,
                obj,
                img_param={'size':size, 'fft':True, 'decorrelate':True},
                img_thres=(n_epochs,),
                dev=dev,
                verbose=False,
                )
        channel_imgs.append(channel_img[0][0])
    channel_imgs = np.array(channel_imgs)
    return channel_imgs



# been used
def prep_v1(paths, model, layer_func, size=(448, 448), dev='cuda:0'):
    # TODO rename
    '''
    
    spritemap of feature visualizations
    for each channel: overall (average) importance
    for each img in paths, we need:
        - image
        - cam-spritemap
        - for each channel: importance
    '''
    # TODO proper resize/cropping as in data set
    imgs = []
    sprites = []
    attrs = []
    As = []
    for p in paths:
        img = Image.open(p).resize(size)
        imgs.append(img)

        timg = ToTensor()(img).view(1, 3, *size)

        cams = prepare_layer_cams(timg, model, layer_func, dev=dev)
        sprite = build_spritemap(cams, border=0, int_format=True)
        sprites.append(sprite)

        attr = channel_attr_binary(timg, model, layer_func, dev=dev)
        attrs.append(attr)
        A = np.zeros(len(attr))
        for k, v in attr.items(): A[k] = v
        As.append(A)
    # average importance for each channel
    As = np.array(As).mean(0)
    #As = dict( (int(i), float(As[i])) for i in As.argsort(0)[::-1] )
    As = [
            {'channel':int(i), 'importance':float(As[i])}
                for i in As.argsort(0)[::-1] ]

    # average importance, images, cam-spritemaps, importances
    return As, imgs, sprites, attrs

#############################################################
######## END EXPORT #########################################
#############################################################

#############################################################
######## START SEMANTIC DICTS  ##############################
#############################################################


# TODO delete
# not used!
def create_semantic_dict(
        img,
        layer_func,
        model,
        dev='cuda:0',
        n_epochs=250,
        layer_size=(1, 3, 224, 224),
        layer_img_path='',
        ):
    if len(img.shape) == 3:
        img = img.view(1, *img.shape)
    #activations = compute_layer(img, model, layer_func, dev)
    #argsort = activations[0].argsort(0, descending=True)
    attributions = channel_attr_binary(img, model, layer_func, torch.ones((1,1)), dev)

    if os.path.isfile(layer_img_path):
        layer_imgs = pickle.load(open(layer_img_path, 'rb'))
    else:
        layer_imgs = prepare_layer(model, layer_func, n_epochs=n_epochs, size=layer_size, dev=dev)
        if layer_img_path:
            pickle.dump(layer_imgs, open(layer_img_path, 'wb'))
    
    return attributions, layer_imgs

#############################################################
######## END SEMANTIC DICTS  ################################
#############################################################

#############################################################
######## START ATTRIBUTION  #################################
#############################################################

# been used
def channel_attr_binary(img, model, layer_func, target_class=torch.ones((1,1)), dev='cuda:0'):
    # TODO --> this dict thing is really weird... -> need it for sorted in vis1?
    act, grad = compute_layer(
            img,
            model,
            layer_func,
            dev,
            include_grad=True,
            targets=torch.tensor(target_class)
            )
    attribution = act * grad[0]
    channel_attr = attribution.sum([-1,-2])
    argsort = channel_attr.argsort(-1, descending=True)
    return dict( (int(s), float(channel_attr[0, s])) for s in argsort[0] )

def channel_cam(img, model, layer_func, channel, target_class=torch.ones((1,1)), dev='cuda:0'):
    act, grad = compute_layer(
            img,
            model,
            layer_func,
            dev,
            include_grad=True,
            targets=torch.tensor(target_class),
            )
    attribution = act * grad[0]
    channel_attr = attribution[0, channel, :, :].detach().cpu().numpy()
    channel_attr = sk_resize(channel_attr, img.shape[-2:], order=0)
    return channel_attr

# been used
def prepare_layer_cams(img, model, layer_func, target_class=torch.ones((1, 1)), dev='cuda:0'):
    normalize = 2

    b, ch, h, w = compute_layer(img, model, layer_func, dev)[0].shape
    cams = []
    for tc in range(ch):
        cam = channel_cam(img, model, layer_func, tc, target_class, dev)
        if (normalize > 1) and (cam.max() - cam.min() > 1e-12):
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        cams.append(cam)
    cams = np.array(cams)
    if normalize == 1:
        cams = (cams - cams.min()) / (cams.max() - cams.min())

    return cams

    


#############################################################
######## END ATTRIBUTION  ###################################
#############################################################

#############################################################
######## START PREP STUFF ###################################
#############################################################


# been used
# TODO probably needs better importance measure
def prep_semantic_dicts(img, model, layer_func, first=0, dev='cuda:0'):
    # TODO how to measure importance? just which activation fired the most now? check with other methods
    '''

    we need:
    - input image
    - activation visualizations of the layer
    - for each position in channel grid: [{layer_n, layer_val} for layer in argsort layers_in_this_pos]

    '''

    act = compute_layer(img, model, layer_func, dev)[0][0]
    values, inds = act.sort(dim=0, descending=True)
    values, inds = values.detach().cpu().numpy().astype(float), inds.detach().cpu().numpy().astype(int)
    if first:
        values, inds = values[:first], inds[:first]

    rows, cols = act.shape[1:]
    #{'pos': {'x': ..., 'y': ...}, 'activations': [{'channel': ..., 'value': ...}, ... ] }

    all_acts = []
    for row in range(rows):
        for col in range(cols):

            loc_dict = {
                'pos': {
                    'x': col,
                    'y': row,
                    },
                'activations': [ {'channel': int(ind), 'value': val} for ind, val in zip(inds[:, row, col], values[:, row, col]) ]
            }
            all_acts.append(loc_dict)
    return all_acts



        








#############################################################
######## END PREP STUFF #####################################
#############################################################




#############################################################
######## START ALPHA ########################################
#############################################################


'''
def to_valid_rgb(img, pre_correlation, post_correlation, decorrelate=False, train=True):
    if len(img) == 2:
        img, alpha = img
        is_alpha = True
    else:
        img = img[0]
        is_alpha = False
    img = pre_correlation(img)
    if decorrelate:
        img = linear_decorrelate(img)
    img = post_correlation(img)

    if is_alpha:
        alpha = post_correlation(alpha)
        if train:
            #background = torch.rand_like(img)
            b, ch, h, w = img.shape
            background = get_bg_img((b, ch, h, w), sd=0.2, decay_power=1.5).to(img.device)
        else:
            background = torch.zeros_like(img)
            #background = torch.ones_like(img)
        img = img * alpha + background * (1 - alpha)
    return img

def get_bg_img(shape, sd=0.2, decay_power=1.5, decorrelate=True):
    b, ch, h, w = shape
    imgs = []
    for _ in range(b):
        freqs = rfft2d_freqs(h, w)
        fh, fw = freqs.shape
        spectrum_var = torch.normal(0, sd, (3, fh, fw, 2))
        scale = np.sqrt(h*w) / np.maximum(freqs, 1./max(h,w))**decay_power

        scaled_spectrum = spectrum_var * torch.from_numpy(scale.reshape(1, *scale.shape, 1)).float()
        img = torch.irfft(scaled_spectrum, signal_ndim=2)
        img = img[:ch, :h, :w]
        imgs.append(img)

    # 4 for desaturation / better scale... 
    imgs = torch.stack(imgs) / 4
    if decorrelate:
        imgs = linear_decorrelate(imgs)
    return imgs.sigmoid()
def get_image(size, std, fft=False, dev='cuda:0', seed=124, decay_power=1., alpha=False):


    if fft:
        b, ch, h, w = size
        freqs = rfft2d_freqs(h, w)
        freq_size = (b, ch, *freqs.shape, 2)
        img = torch.normal(0, std, freq_size).to(dev).requires_grad_()

        scale = (torch.tensor(np.sqrt(w*h) / np.maximum(freqs, 1./max(w, h)) ** decay_power))
        scale = scale.view(1, 1, *scale.shape, 1).float().to(dev)
        def pre_correlation(img):
            scaled_img = scale * img
            rgb_img = torch.irfft(scaled_img, signal_ndim=2)
            rgb_img = rgb_img[:b, :ch, :h, :w]
            # TODO what? why? maybe approx the same scale as for pixel images?
            rgb_img.div_(4)
            return rgb_img

        def post_correlation(img):
            return torch.sigmoid(img)
        
    else:
        img = torch.normal(0, std, size).to(dev).requires_grad_()
        pre_correlation = lambda x: x
        def post_correlation(img):
            return torch.sigmoid(img)
    if alpha:
        alpha_shape = (size[0], 1, *size[2:])
        img_alpha = torch.normal(0, std, size).to(dev).requires_grad_()
        return [img, img_alpha], pre_correlation, post_correlation
    else:
        return [img], pre_correlation, post_correlation

class ImageObjective(Objective):
    def __init__(self):
        super().__init__(None)

    def register(self, model):
        pass

    def remove_hook(self):
        pass

class AlphaObjective(ImageObjective):
    def __init__(self, alpha_channels):
        super().__init__()
        self.alpha_channels = alpha_channels
    
    def _compute_loss(self):
        return 1-self.alpha_channels.sigmoid().mean([-1,-2,-3])
''' 
#############################################################
######## END ALPHA ##########################################
#############################################################


