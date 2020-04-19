import warnings
from os.path import join

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import pearsonr
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from tqdm import tqdm

from skimage.transform import resize as sk_resize
from utils import prep_model

warnings.filterwarnings('ignore', category=UserWarning)


def get_channel_to_output_correlation(model, dataset, layer_func, n_channels=5, n_loader=2, dev='cuda'):
    model = prep_model(model, dev)
    output = []

    def layer_hook(i, m, o):
        output.append(o[:, :, :, :].detach().cpu().sum([-1, -2]))
    hook = layer_func(model).register_forward_hook(layer_hook)
    preds = []
    for idx, inp in tqdm(enumerate(dataset)):
        if n_loader > 1:
            inp = inp[0]
        inp = inp.to(dev).view(1, *inp.shape)
        pred = model(inp).item()
        preds.append(pred)
    hook.remove()
    output = torch.cat(output).numpy()
    preds = np.array(preds)
    correlations = np.array([pearsonr(preds, output[:, i])[0]
                             for i in range(output.shape[1])])
    cc_sort = np.argsort(correlations)
    max_correlation = cc_sort[-n_channels:][::-1]
    min_correlation = cc_sort[:n_channels]
    return output, preds, correlations, min_correlation, max_correlation


def get_img_to_channel(model, dataset, layer_func, channel=0, n_imgs=3, n_loader=2, dev='cuda'):
    model = prep_model(model, dev)
    output = []

    def channel_hook(i, m, o):
        output.append(o[:, channel, :, :].detach().cpu())
    hook = layer_func(model).register_forward_hook(channel_hook)
    for idx, inp in tqdm(enumerate(dataset)):
        if n_loader > 1:
            inp = inp[0]
        inp = inp.to(dev).view(1, *inp.shape)
        model(inp).item()
    hook.remove()
    output_reduced = [o.sum([-1, -2]) for o in output]
    output_reduced = torch.cat(output_reduced)
    ind = list(output_reduced.argsort()[-n_imgs:].numpy())[::-1]
    imgs = []
    for idx in ind:
        img, target, _ = dataset.get_before_tensor(idx)
        val = output_reduced[idx].item()
        heatmap = sk_resize(output[idx][0], img.size[-2:])
        imgs.append((img, target, val, heatmap))

    return imgs


def get_patch_to_channel(model, dataset, layer_func, channel=0, n_patches=3, n_loader=2, dev='cuda'):
    model = prep_model(model, dev)
    output = []

    def channel_hook(i, m, o):
        output.append(o[:, channel, :, :].detach().cpu())
    hook = layer_func(model).register_forward_hook(channel_hook)
    for idx, inp in tqdm(enumerate(dataset)):
        if n_loader > 1:
            inp = inp[0]
        inp = inp.to(dev).view(1, *inp.shape)
        model(inp).item()
    output = torch.cat(output)

    full_x, full_y = inp.shape[-2:]
    layer_x, layer_y = output.shape[-2:]
    pix_per_box_x, pix_per_box_y = full_x // layer_x, full_y // layer_y
    print(full_x, layer_x, pix_per_box_x)

    output = [(output[b, i, j].item(), b, i, j)
              for b in range(output.shape[0])
              for i in range(output.shape[1])
              for j in range(output.shape[2])]
    output = sorted(output)
    patches = []
    for val, b, i, j in output[-n_patches:][::-1]:
        x, y = i * layer_x, j * layer_y
        img = dataset.get_before_tensor(b)
        if n_loader > 1:
            img = img[0]
        patch = img.crop(box=(y, x, y + pix_per_box_y, x + pix_per_box_x))
        # patch = img.crop(box=(x, y, x+pix_per_box_x, y+pix_per_box_y))
        patches.append((patch, val))

    hook.remove()
    return patches


def get_highest_activation_images(model, dataset, n_imgs=3, pred_ind=None, n_loader=2, dev='cuda'):
    '''

    # Parameters
    model
    dataset
    pred_ind (None or int): which index of output to use. if int, assumes that output of model has size (batch_size, n_output) and 0 <= pred_ind < n_output
    n_loader (int): number of outputs of the loader - usually 2 for (imgs, targets) pairs, but sometimes more/less
    dev
    '''
    model = prep_model(model, dev)
    all_out = []
    for idx, inp in tqdm(enumerate(dataset)):
        if n_loader > 1:
            inp = inp[0]
        inp = inp.to(dev).view(1, *inp.shape)
        out = model(inp)[:, pred_ind].flatten().item()
        all_out.append(out)
    # sort_idx = np.argsort(all_out)
    # lo_idx, hi_idx = sort_idx[:n_imgs], sort_idx[-n_imgs:]
    return all_out


# TODO TMP -- for testing purposes only atm
def get_data(size_1=512, size_2=None, size=None, n_subs=1000):
    BASE_IMG = '/home/Matthias.Kirchler/retina/kaggle/data/'
    train_dir = 'train_max512'
    df = pd.read_csv(join(BASE_IMG, 'trainLabels.csv'))
    df.image = [join(BASE_IMG, train_dir, p + '.jpeg') for p in df.image]
    df = df.drop([1146, ])[:n_subs]

    if size_1 is None:
        size_1 = 512
    if size_2 is None:
        size_2 = size_1
    if size is None:
        size = size_2

    tfms = transforms.Compose([
        transforms.CenterCrop(size_1),
        transforms.Resize(size_2, interpolation=3),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dset = ImageDataset(df, tfms=tfms, target='level')
    return dset


class ImageDataset(Dataset):
    def __init__(self, df, tfms=None, target=None):
        self.df = df
        self.tfms = tfms
        self.target = target

        names = [tfm.__str__() for tfm in tfms.transforms]
        to_tensor_idx = names.index('ToTensor()')
        self.tfms_before_tensor = transforms.Compose(
            tfms.transforms[:to_tensor_idx])

    def __len__(self):
        return len(self.df)

    def get_before_tensor(self, idx):
        # print(idx, type(idx))
        path = self.df.iloc[idx].image
        orig_img = Image.open(path)
        if self.tfms is not None:
            img = self.tfms_before_tensor(orig_img)

        if self.target is not None:
            target = self.df.iloc[idx][self.target]
            return img, target, orig_img
        else:
            return img, orig_img

    def __getitem__(self, idx):
        path = self.df.iloc[idx].image
        img = Image.open(path)
        if self.tfms is not None:
            img = self.tfms(img)

        if self.target is not None:
            target = self.df.iloc[idx][self.target]
            return img, target
        else:
            return img


D = get_data(512, 512 + 128, 448, n_subs=100)


#####
