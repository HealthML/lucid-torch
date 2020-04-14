import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from functools import partial
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from torch import optim
from torchvision import models

from img_param import *
from objectives import *
from transforms import *
from utils import *

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
