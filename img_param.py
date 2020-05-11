from enum import Enum, auto

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

# from lucid (on imagenet?)
color_correlation_svd_sqrt = torch.tensor([[0.26, 0.09, 0.02],
                                           [0.27, 0.00, -0.05],
                                           [0.27, -0.09, 0.03]])

# from ukb retina data (but is it correct?)
color_correlation_svd_sqrt_custom = torch.tensor([[0.179, -0.041, -0.005],
                                                  [0.100, 0.049, 0.016],
                                                  [0.042, 0.057, -0.017]])

# TODO check - is this how color_correlation_svd_sqrt is meant to be?
# (eg try computing on large imagenet sample)
# data = ... # (n_samples, 3, w, h)
# B = data.mean([-1,-2])
# cov = np.cov(B.numpy().T)
# u, s, vt = np.linalg.svd(cov)
# color_correlation_svd_sqrt = u @ np.diag(s**0.5)
C = color_correlation_svd_sqrt
# C = color_correlation_svd_sqrt_custom
max_norm_svd_sqrt = torch.norm(C, dim=0).max()
cc_norm = C / max_norm_svd_sqrt


def init_from_image(paths, size=(224, 224), fft=False, dev='cuda:0', eps=1e-4):
    if not isinstance(paths, list):
        paths = [paths]
    imgs = []
    for path in paths:
        # TODO refactor into more flexible load_img fct
        img = Image.open(path).convert('RGB').resize(size)
        img = ToTensor()(img).view(1, 3, *size)
        img[img == 1] = 1. - eps
        img[img == 0] = eps

        img = torch.log(img / (1. - img)).to(dev)

        if fft:
            w, h = size
            freqs = rfft2d_freqs(*size)
            decay_power = 1
            scale = (torch.tensor(np.sqrt(w * h) /
                                  np.maximum(freqs, 1. / max(w, h)) ** decay_power))
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

        def post_correlation(x):
            return torch.sigmoid(x)
    else:
        def pre_correlation(x):
            return x

        def post_correlation(x):
            return torch.sigmoid(x)

    return imgs, pre_correlation, post_correlation


def get_image(size, std, fft=False, dev='cuda:0', seed=124, decay_power=1., alpha=False):
    if fft:
        b, ch, h, w = size
        freqs = rfft2d_freqs(h, w)
        freq_size = (b, ch, *freqs.shape, 2)
        img = torch.normal(0, std, freq_size).to(dev).requires_grad_()

        scale = (torch.tensor(np.sqrt(w * h) /
                              np.maximum(freqs, 1. / max(w, h)) ** decay_power))
        scale = scale.view(1, 1, *scale.shape, 1).float().to(dev)

        def pre_correlation(img):
            scaled_img = scale * img
            rgb_img = torch.irfft(scaled_img, signal_ndim=2)
            rgb_img = rgb_img[:b, :ch, :h, :w]
            # TODO why? approx the same scale as for pixel images, but is there really a reason?
            rgb_img.div_(4)
            return rgb_img

        def post_correlation(img):
            return torch.sigmoid(img)

    else:
        img = torch.normal(0, std, size).to(dev).requires_grad_()

        def pre_correlation(x):
            return x

        def post_correlation(img):
            return torch.sigmoid(img)
    if alpha:
        alpha_shape = (size[0], 1, *size[2:])

        img_alpha = torch.normal(-0.8, std,
                                 alpha_shape).to(dev).requires_grad_()
        return [img, img_alpha], pre_correlation, post_correlation
    else:
        return [img], pre_correlation, post_correlation


def rfft2d_freqs(h, w):
    '''compute 2D spectrum frequencies (from lucid)'''
    fy = np.fft.fftfreq(h)[:, None]
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[:((w // 2) + 2)]
    else:
        fx = np.fft.fftfreq(w)[:((w // 2) + 1)]
    return np.sqrt(fx * fx + fy * fy)


class BackgroundStyle(Enum):
    BLACK = auto()
    WHITE = auto()
    RAND = auto()
    RAND_FREQ = auto()


def to_valid_rgb(img, pre_correlation, post_correlation, decorrelate=False, background=BackgroundStyle.RAND_FREQ):
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
        alpha = alpha.sigmoid()
        if background is BackgroundStyle.BLACK:
            background = torch.zeros_like(img)
        elif background is BackgroundStyle.WHITE:
            background = torch.ones_like(img)
        elif background is BackgroundStyle.RAND:
            background = torch.rand_like(img)
        elif background is BackgroundStyle.RAND_FREQ:
            b, ch, h, w = img.shape
            background = get_bg_img(
                (b, ch, h, w), sd=0.2, decay_power=1.5).to(img.device)
        else:
            raise NotImplementedError
        img = img * alpha + background * (1 - alpha)
    return img


def linear_decorrelate(img):
    '''multiply input by sqrt of empirical ImageNet color correlation matrix'''
    img = img.permute(0, 2, 3, 1)
    img = (img.reshape(-1, 3) @ cc_norm.t().to(img.device)).view(*img.shape)
    img = img.permute(0, 3, 1, 2)
    return img


def get_bg_img(shape, sd=0.2, decay_power=1.5, decorrelate=True):
    b, ch, h, w = shape
    imgs = []
    for _ in range(b):
        freqs = rfft2d_freqs(h, w)
        fh, fw = freqs.shape
        spectrum_var = torch.normal(0, sd, (3, fh, fw, 2))
        scale = np.sqrt(h * w) / np.maximum(freqs, 1. / max(h, w))**decay_power

        scaled_spectrum = spectrum_var * \
            torch.from_numpy(scale.reshape(1, *scale.shape, 1)).float()
        img = torch.irfft(scaled_spectrum, signal_ndim=2)
        img = img[:ch, :h, :w]
        imgs.append(img)

    # 4 for desaturation / better scale...
    imgs = torch.stack(imgs) / 4
    if decorrelate:
        imgs = linear_decorrelate(imgs)
    return imgs.sigmoid()
