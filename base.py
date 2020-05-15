import warnings
from functools import partial

import numpy as np
import torch
from matplotlib import pyplot as plt
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from torch import optim
from tqdm import tqdm

from img_param import BackgroundStyle, get_image, init_from_image, to_valid_rgb
from transforms import (TFMSAlpha, TFMSJitter, TFMSNormalize, TFMSPad,
                        TFMSRandomGaussBlur, TFMSRandomRotate, TFMSRandomScale)
from utils import prep_model

warnings.filterwarnings('ignore', category=UserWarning)


def img_from_param(size=(1, 3, 128, 128), std=0.01, fft=True, decorrelate=True, decay_power=1, seed=42, dev='cpu', path=None, eps=1e-5, alpha=False):
    if path is not None:
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
            alpha=alpha
        )
    to_rgb = partial(to_valid_rgb,
                     pre_correlation=pre, post_correlation=post, decorrelate=decorrelate)
    return img, to_rgb


def opt_from_param(img, opt='adam', lr=0.05, eps=1e-7, wd=0.):
    if opt == 'adam':
        opt = optim.Adam(img, lr=lr, eps=1e-7, weight_decay=wd)
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

    return torch.nn.Sequential(*tfm_param)


def alpha_tfms_from_param(alpha_tfm_param):
    if alpha_tfm_param is None:
        return None
    if isinstance(alpha_tfm_param, str) and alpha_tfm_param == 'default':
        alpha_tfm_param = [
            TFMSRandomGaussBlur((13, 13), (31, 31),
                                (5, 5), (17, 17), border_type='constant')
        ]
    elif not isinstance(alpha_tfm_param, list):
        raise NotImplementedError

    return TFMSAlpha(torch.nn.Sequential(*alpha_tfm_param))


def render(model, objective, img_thres=(100,),
           img_param={}, opt_param={}, tfm_param='default',
           alpha_tfm_param='default',
           seed=None, dev='cuda:0',
           verbose=True,
           video=None
           ):
    if seed:
        torch.manual_seed(seed)

    model = prep_model(model, dev)
    img, to_rgb = img_from_param(dev=dev, **img_param)
    opt = opt_from_param(img, **opt_param)
    tfms = tfms_from_param(tfm_param)
    alpha_tfms = alpha_tfms_from_param(alpha_tfm_param)
    objective.register(model, img)

    imgs = []
    video = open_video(video, img_param['size'][2:])
    if verbose:
        pbar = tqdm(range(1, max(img_thres) + 1))
    else:
        pbar = range(1, max(img_thres) + 1)

    def render_steps():
        for i in pbar:
            step(img, opt, objective, model, to_rgb, tfms, alpha_tfms)
            if verbose:
                with torch.no_grad():
                    pbar.set_description("Epoch %d, current loss: %.3f" % (
                        i, objective._compute_loss()))
            if video is not None:
                frame = to_rgb(
                    img, background=BackgroundStyle.WHITE).detach().cpu().numpy()
                video.write_frame(
                    np.uint8(np.moveaxis(frame, 1, -1)[-1] * 255.0))
            if i in img_thres:
                imgs.append(
                    to_rgb(img, background=BackgroundStyle.WHITE).detach().cpu().numpy())

    if video is None:
        render_steps()
    else:
        with video:
            render_steps()

    imgs = np.moveaxis(np.array(imgs), 2, -1)
    if verbose:
        plot_imgs(imgs[-1])

    objective.remove_hook()
    return imgs


def open_video(video=None, size=(224, 224)):
    if video is not None:
        if isinstance(video, str):
            video = FFMPEG_VideoWriter(video, size, 60.0)
        elif not isinstance(video, FFMPEG_VideoWriter):
            raise NotImplementedError
    return video


def plot_imgs(imgs):
    '''plot imgs into joint figure

    imgs: (batch, height, width, channels)
    '''
    n_img = imgs.shape[0]
    if n_img < 13:
        n_rows = [None, 1, 1, 2, 2, 2, 2,
                  3, 3, 3, 3, 3, 3][n_img]
    else:
        n_rows = 4
    n_cols = np.ceil(n_img / n_rows).astype(int)
    fig = plt.figure(figsize=(10, 10))
    for i, img in enumerate(imgs):
        fig.add_subplot(n_rows, n_cols, i + 1)
        plt.imshow(img)


def step(img, opt, obj, model, to_rgb, tfms, alpha_tfms):
    opt.zero_grad()
    if alpha_tfms is not None:
        img = alpha_tfms(img)
    # e.g. sigmoid, or inverse fft
    img = to_rgb(img)
    if tfms is not None:
        img = tfms(img)
    _ = model(img)
    obj.backward()
    opt.step()
