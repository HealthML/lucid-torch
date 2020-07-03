import torch

from .alpha import BackgroundStyle, TFMSAddAlphaChannel, TFMSMixinAlphaChannel
from .channels import TFMSTransformChannels
from .decorrelate import TFMSDecorrelate
from .fft import TFMSFFT, TFMSIFFT
from .monochrome import TFMSMonochromeTo, TFMSToMonochrome
from .stability import TFMSJitter, TFMSNormalize, TFMSPad, TFMSRandomScale
from .stability.blur import TFMSRandomGaussianBlur
from .stability.rotate import TFMSRandomRotate
from .unit_space.TFMSTrainingToUnitSpace import TFMSTrainingToUnitSpace
from .unit_space.TFMSUnitToTrainingSpace import TFMSUnitToTrainingSpace

DEFAULT_USE_FFT = True
DEFAULT_USE_ALPHA = False
DEFAULT_USE_MONOCHROME = False


def dataspaceTFMS(fft: bool = DEFAULT_USE_FFT,
                  alpha: bool = DEFAULT_USE_ALPHA,
                  monochrome: bool = DEFAULT_USE_MONOCHROME):
    transforms = []
    if monochrome:
        transforms.append(TFMSToMonochrome())
    if alpha:
        transforms.append(TFMSAddAlphaChannel())

    transforms.append(TFMSUnitToTrainingSpace())

    if fft:
        transforms.append(TFMSFFT())
    return torch.nn.Sequential(*transforms)


def drawTFMS(fft: bool = DEFAULT_USE_FFT,
             alpha: bool = DEFAULT_USE_ALPHA,
             monochrome: bool = DEFAULT_USE_MONOCHROME):
    transforms = []
    if fft:
        transforms.append(TFMSIFFT())
    transforms.append(TFMSDecorrelate())
    transforms.append(TFMSTrainingToUnitSpace())
    if alpha:
        transforms.append(TFMSMixinAlphaChannel(BackgroundStyle.WHITE))
    if monochrome:
        transforms.append(TFMSMonochromeTo())
    return torch.nn.Sequential(*transforms)


def trainTFMS(fft: bool = DEFAULT_USE_FFT,
              alpha: bool = DEFAULT_USE_ALPHA,
              monochrome: bool = DEFAULT_USE_MONOCHROME,
              normalize: bool = True):
    transforms = []
    if fft:
        transforms.append(TFMSIFFT())
    transforms.append(TFMSDecorrelate())
    transforms.append(TFMSTrainingToUnitSpace())
    if alpha:
        alpha_transforms = [TFMSRandomGaussianBlur((13, 13),
                                                   (31, 31),
                                                   (5, 5),
                                                   (17, 17),
                                                   border_type='constant')]
        transforms.append(TFMSTransformChannels(-1,
                                                torch.nn.Sequential(*alpha_transforms)))
        transforms.append(TFMSMixinAlphaChannel(BackgroundStyle.RAND_FFT))
    if monochrome:
        transforms.append(TFMSMonochromeTo())

    transforms.extend([
        TFMSPad(12, 'constant', 0.5),
        TFMSJitter(8),
        TFMSRandomScale([1 + (i - 5) / 50. for i in range(11)]),
        TFMSRandomRotate(list(range(-10, 11)) + 5 * [0]),
        TFMSJitter(4)
    ])
    if normalize:
        transforms.append(TFMSNormalize())
    return torch.nn.Sequential(*transforms)
