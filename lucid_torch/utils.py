import numpy as np
import torch
from torch import nn


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
    '''compute intermediate output at layer layer_func

    if include_grad, also compute gradient wrt targets; currently only works with scalar targets

    m = model
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
    hook = layer_func(model).register_forward_hook(
        lambda m, i, o: output.append(o.detach()))
    if include_grad:
        hook_backward = layer_func(model).register_backward_hook(
            lambda m, i, o: output.append(o))

    out = model(inp.to(dev))

    hook.remove()
    if include_grad:
        loss = nn.BCELoss()(torch.sigmoid(out), targets.to(dev))
        loss.backward()
        hook_backward.remove()

    return output


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
        sprite_size = ((n_rows - 1) * (h + border) + h,
                       (n_cols - 1) * (w + border) + w, 3)
    else:
        sprite_size = ((n_rows - 1) * (h + border) + h,
                       (n_cols - 1) * (w + border) + w)
    sprite = np.ones(sprite_size)
    i = j = 0
    for img in channel_imgs:
        hh, hp = i * (h + border), i * (h + border) + h
        ww, wp = j * (w + border), j * (w + border) + w
        sprite[hh:hp, ww:wp] = img
        j = (j + 1) % n_cols
        if j == 0:
            i += 1
    if int_format:
        sprite = np.round(sprite * 255).astype(np.uint8)
    return sprite


def namify(string):
    return '_'.join(string.split())
