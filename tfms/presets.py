import torch

from tfms.alpha import (BackgroundStyle, TFMSAddAlphaChannel,
                        TFMSMixinAlphaChannel)
from tfms.channel import TFMSTransformChannels
from tfms.decorrelate import TFMSDecorrelate
from tfms.fft import TFMSFFT, TFMSIFFT
from tfms.monochrome import TFMSMonochromeTo, TFMSToMonochrome
from tfms.stability import (TFMSJitter, TFMSNormalize, TFMSPad,
                            TFMSRandomGaussBlur, TFMSRandomRotate,
                            TFMSRandomScale)
from tfms.unit_space import TFMSTrainingToUnitSpace, TFMSUnitToTrainingSpace

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
        tfms.append(TFMSAddAlphaChannel())

    tfms.append(TFMSUnitToTrainingSpace())

    if fft:
        tfms.append(TFMSFFT())
    return torch.nn.Sequential(*tfms)


def drawTFMS(fft: bool = DEFAULT_USE_FFT,
             alpha: bool = DEFAULT_USE_ALPHA,
             monochrome: bool = DEFAULT_USE_MONOCHROME):
    tfms = []
    if fft:
        tfms.append(TFMSIFFT())
    tfms.append(TFMSDecorrelate())
    tfms.append(TFMSTrainingToUnitSpace())
    if alpha:
        tfms.append(TFMSMixinAlphaChannel(BackgroundStyle.BLACK))
        # TODO_M Das Bild hat einen fast gleichmäßigen Alpha Kanal?
        # Wenn wir hier RAND_FFT nutzen, geht es erstaunlicherweise einigermaßen gut
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
    tfms.append(TFMSDecorrelate())
    tfms.append(TFMSTrainingToUnitSpace())
    if alpha:
        alpha_tfms = [TFMSRandomGaussBlur((13, 13),
                                          (31, 31),
                                          (5, 5),
                                          (17, 17),
                                          border_type='constant')]
        tfms.append(TFMSTransformChannels(-1,
                                          torch.nn.Sequential(*alpha_tfms)))
        tfms.append(TFMSMixinAlphaChannel(BackgroundStyle.RAND_FFT))
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
