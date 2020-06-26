import torch

from transforms.alpha.BackgroundStyle import BackgroundStyle
from transforms.alpha.TFMSAddAlphaChannel import TFMSAddAlphaChannel
from transforms.alpha.TFMSMixinAlphaChannel import TFMSMixinAlphaChannel
from transforms.channels.TFMSTransformChannels import TFMSTransformChannels
from transforms.decorrelate.TFMSDecorrelate import TFMSDecorrelate
from transforms.fft.TFMSFFT import TFMSFFT
from transforms.fft.TFMSIFFT import TFMSIFFT
from transforms.monochrome.TFMSMonochromeTo import TFMSMonochromeTo
from transforms.monochrome.TFMSToMonochrome import TFMSToMonochrome
from transforms.stability.TFMSJitter import TFMSJitter
from transforms.stability.TFMSNormalize import TFMSNormalize
from transforms.stability.TFMSPad import TFMSPad
from transforms.stability.blur.TFMSRandomGaussianBlur import TFMSRandomGaussBlur
from transforms.stability.rotate.TFMSRandomRotate import TFMSRandomRotate
from transforms.stability.TFMSRandomScale import TFMSRandomScale
from transforms.unit_space.TFMSTrainingToUnitSpace import TFMSTrainingToUnitSpace
from transforms.unit_space.TFMSUnitToTrainingSpace import TFMSUnitToTrainingSpace

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
        transforms.append(TFMSMixinAlphaChannel(BackgroundStyle.BLACK))
        # TODO_M Das Bild hat einen fast gleichmäßigen Alpha Kanal?
        # Wenn wir hier RAND_FFT nutzen, geht es erstaunlicherweise einigermaßen gut
        # Problem könnte daran liegen, dass alpha mit in fft space ist
        # (kann nicht so leicht geändert werden, da fft space die dimensionen verändert)
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
        alpha_transforms = [TFMSRandomGaussBlur((13, 13),
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
