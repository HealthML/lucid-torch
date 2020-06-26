import json
import os
from os.path import join

import numpy as np
import torch
from matplotlib.pyplot import imsave
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm

from attribution import channel_attr_binary, prepare_layer_cams
from base import render
from objectives.channel.ChannelObjective import ChannelObjective
from objectives.neuron.FCNeuronObjective import FCNeuronObjective
from utils import build_spritemap, compute_layer, namify


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
    selected_paths = list(np.random.choice(
        image_paths, size=n_images, replace=False))

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
        layer_fvs = prepare_layer(
            model, layer_func, n_epochs=n_epochs, size=(1, 3, *size), dev=dev)
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
    path_to_imgs = [join(dir_name, 'sample_img%d.jpg' % d)
                    for d in range(len(imgs))]
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
    path_to_imgs = [join(dir_name, 'sample_img%d.jpg' % d)
                    for d in range(n_images_vis_4)]
    for out_p, in_p in zip(path_to_imgs, selected_paths[:n_images_vis_4]):
        # TODO better resize
        img = Image.open(in_p).convert('RGB').resize(size)
        img.save(out_p)


def save_v4(paths, model, layer_func, path_to_json, first=3, size=(448, 448), dev='cuda:0'):
    for i, p in enumerate(paths):
        img = load_img(p, size, dev)
        sem_dict = prep_semantic_dicts(
            img, model, layer_func, first=first, dev=dev)
        with open(path_to_json % i, 'w') as f:
            json.dump(sem_dict, f)


def prep_v2(model, n_epochs, paths, all_paths, size=(448, 448), dev='cuda:0'):
    # TODO rename
    def final_layer_func(m):
        return m
    obj = FCNeuronObjective(final_layer_func, neuron=0)
    n_images = len(paths)

    # basic fv
    pos_act = render(model, obj, img_thres=(n_epochs,),
                     img_param={'size': (n_images, 3, *size)},
                     dev=dev,
                     verbose=False,
                     )
    neg_act = render(model, -obj, img_thres=(n_epochs,),
                     img_param={'size': (n_images, 3, *size)},
                     dev=dev,
                     verbose=False,
                     )

    # fv from image init
    img_param_init = {'size': (n_images, 3, *size),
                      'fft': False, 'decorrelate': False, 'path': paths}
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
    pos_act_imgs = np.array(
        [np.array(Image.open(all_paths[i]).resize(size)) for i in hi_act_ind]) / 255.
    neg_act_imgs = np.array(
        [np.array(Image.open(all_paths[i]).resize(size)) for i in lo_act_ind]) / 255.

    pos_act, neg_act = build_spritemap(
        pos_act[0], int_format=True), build_spritemap(neg_act[0], int_format=True)
    pos_act_init, neg_act_init = build_spritemap(
        pos_act_init[0], int_format=True), build_spritemap(neg_act_init[0], int_format=True)
    pos_act_imgs, neg_act_imgs = build_spritemap(
        pos_act_imgs, int_format=True), build_spritemap(neg_act_imgs, int_format=True)
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

    path_to_sprites = [join(sdir, 'cam_sprite%d.jpg' % d)
                       for d in range(len(sprites))]
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
    # with open(path_to_pos_act, 'w') as f:
    imsave(path_to_pos_act, pos_act)

    path_to_neg_act = join(sdir, 'neg_act.%s' % img_extension)
    # with open(path_to_neg_act, 'w') as f:
    imsave(path_to_neg_act, neg_act)

    path_to_pos_act_init = join(sdir, 'pos_act_init.%s' % img_extension)
    # with open(path_to_pos_act_init, 'w') as f:
    imsave(path_to_pos_act_init, pos_act_init)

    path_to_neg_act_init = join(sdir, 'neg_act_init.%s' % img_extension)
    # with open(path_to_neg_act_init, 'w') as f:
    imsave(path_to_neg_act_init, neg_act_init)

    path_to_pos_act_img = join(sdir, 'pos_act_img.%s' % img_extension)
    # with open(path_to_pos_act_img, 'w') as f:
    imsave(path_to_pos_act_img, pos_act_imgs)

    path_to_neg_act_img = join(sdir, 'neg_act_img.%s' % img_extension)
    # with open(path_to_neg_act_img, 'w') as f:
    imsave(path_to_neg_act_img, neg_act_imgs)


# been used in vis
def prepare_layer(model, layer_func, n_epochs=250, size=(1, 3, 224, 224), dev='cuda:0'):
    # TODO rename
    '''compute feature visualizations for each channel in layer'''
    _, ch, _, _ = compute_layer(torch.normal(
        0, 0.01, size), model, layer_func, dev)[0].shape

    channel_imgs = []
    for c in tqdm(range(ch)):
        obj = ChannelObjective(layer_func, channel=c)
        channel_img = render(
            model,
            obj,
            img_param={'size': size, 'fft': True, 'decorrelate': True},
            img_thres=(n_epochs,),
            dev=dev,
            verbose=False,
        )
        channel_imgs.append(channel_img[0][0])
    channel_imgs = np.array(channel_imgs)
    return channel_imgs


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
        for k, v in attr.items():
            A[k] = v
        As.append(A)
    # average importance for each channel
    As = np.array(As).mean(0)
    # As = dict( (int(i), float(As[i])) for i in As.argsort(0)[::-1] )
    As = [
        {'channel': int(i), 'importance': float(As[i])}
        for i in As.argsort(0)[::-1]]

    # average importance, images, cam-spritemaps, importances
    return As, imgs, sprites, attrs


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
    values, inds = values.detach().cpu().numpy().astype(
        float), inds.detach().cpu().numpy().astype(int)
    if first:
        values, inds = values[:first], inds[:first]

    rows, cols = act.shape[1:]
    # {'pos': {'x': ..., 'y': ...}, 'activations': [{'channel': ..., 'value': ...}, ... ] }

    all_acts = []
    for row in range(rows):
        for col in range(cols):

            loc_dict = {
                'pos': {
                    'x': col,
                    'y': row,
                },
                'activations': [{'channel': int(ind), 'value': val} for ind, val in zip(inds[:, row, col], values[:, row, col])]
            }
            all_acts.append(loc_dict)
    return all_acts
