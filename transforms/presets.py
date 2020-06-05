from alpha import TFMSAddAlphaChannel, TFMSMixinAlphaChannel, BackgroundStyle
from monochrome import TFMSMonochromeTo, TFMSToMonochrome
from fft import TFMSFFT, TFMSIFFT
import torch
from channel import TFMSTransformChannels
from stability import (TFMSJitter, TFMSNormalize, TFMSPad,
                       TFMSRandomGaussBlur, TFMSRandomRotate, TFMSRandomScale)

DEFAULT_USE_FFT = True
DEFAULT_USE_ALPHA = False
DEFAULT_USE_MONOCHROME = False


def dataspaceTFMS(fft: bool = DEFAULT_USE_FFT,
                  alpha: bool = DEFAULT_USE_ALPHA,
                  monochrome: bool = DEFAULT_USE_MONOCHROME):
    tfms = []
    if monochrome:
        tfms.append(TFMSToMonochrome())
    if alpha:
        tfms.append(TFMSAddAlphaChannel)
    if fft:
        tfms.append(TFMSFFT())
    return torch.nn.Sequential(*tfms)


def drawTFMS(fft: bool = DEFAULT_USE_FFT,
             alpha: bool = DEFAULT_USE_ALPHA,
             monochrome: bool = DEFAULT_USE_MONOCHROME):
    tfms = []
    if fft:
        tfms.append(TFMSIFFT())
    if alpha:
        tfms.append(TFMSMixinAlphaChannel(BackgroundStyle.WHITE))
    if monochrome:
        tfms.append(TFMSMonochromeTo())
    return torch.nn.Sequential(*tfms)


def trainTFMS(fft: bool = DEFAULT_USE_FFT,
              alpha: bool = DEFAULT_USE_ALPHA,
              monochrome: bool = DEFAULT_USE_MONOCHROME,
              normalize: bool = False):
    tfms = []
    if fft:
        tfms.append(TFMSIFFT())
    if alpha:
        alpha_tfms = [TFMSRandomGaussBlur((13, 13),
                                          (31, 31),
                                          (5, 5),
                                          (17, 17),
                                          border_type='constant')]
        tfms.append(TFMSTransformChannels(-1, torch.nn.Sequential(alpha_tfms)))
        tfms.append(TFMSMixinAlphaChannel(BackgroundStyle.WHITE))
    if monochrome:
        tfms.append(TFMSMonochromeTo())

    tfms.extend([
        TFMSPad(12, 'constant', 0.5),
        TFMSJitter(8),
        TFMSRandomScale([1 + (i - 5) / 50. for i in range(11)]),
        TFMSRandomRotate(list(range(-10, 11)) + 5 * [0]),
        TFMSJitter(4)
    ])
    if normalize:
        tfms.append(TFMSNormalize())
    return torch.nn.Sequential(*tfms)
