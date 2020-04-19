from base import *


def basic(dev='cuda:0'):
    M = models.resnet18(pretrained=True)

    # look into 3rd layer, 10th channel; this changes a lot between architectures
    obj = Channel(lambda m: m.layer3[1].conv2, channel=15)

    # return 4 visualizations of size 224 x 224
    size = (4, 3, 224, 224)
    # fft & decorrelate usually look a little nicer than not
    # other params less important; gets passed to img_from_param
    img_param = {'size': size, 'fft': True, 'decorrelate': True}

    # what kind of data augmentation?
    # if model was trained with normalization, use 'default_norm' or list with corresponding normalization, otherwise 'default'
    tfm_param = 'default_norm'

    # render is the full interface for a user
    imgs = render(
        M,                  # always specify model
        obj,                # always specify objective
        img_thres=(1000,),  # run for 1000 steps; needs to be tuple/list,
        # if more than one value, return intermediate results as well
        img_param=img_param,
        tfm_param=tfm_param,
        dev=dev,
        # will make progress bar and directly plot the imgs if on notebook or x-server active
        verbose=True,
    )
    return imgs


def objective(dev='cuda:0'):
    M = models.resnet18(pretrained=True)

    # visualize full channel
    obj1 = Channel(lambda m: m.layer3[1].conv2, channel=15)
    # visualize single (center) neuron in this channel
    obj2 = ConvNeuron(lambda m: m.layer3[1].conv2, channel=15)
    # visualize neuron in fully connected layer, in this case already output
    obj3 = FCNeuron(lambda m: m.fc, neuron=123)

    # objectives can be combined
    obj4 = 0.5 * obj1 + obj2 - obj3

    size = (4, 3, 224, 224)
    img_param = {'size': size, 'fft': True, 'decorrelate': True}
    tfm_param = 'default_norm'

    imgs = render(
        M,
        obj4,
        img_thres=(1000,),
        img_param=img_param,
        tfm_param=tfm_param,
        dev=dev,
        verbose=True,
    )
    return imgs
