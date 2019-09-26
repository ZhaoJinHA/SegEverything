#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks
import torch
import os

import numpy as np
from PIL import Image

from .utils import resize_and_crop, resize_and_crop_masks, normalize, hwc_to_chw, get_square
from torch.utils import data

def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i)  for id in ids for i in range(n))


def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + '/' + id + suffix), scale=scale)
        if len(im.shape) == 2:
            im = im[...,np.newaxis]
        # print('im.shape', im.shape)
        yield get_square(im, pos)

def to_cropped_masks(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        filename = dir + '/' + id + suffix
        maskarray = np.load(filename)['label']
        im = resize_and_crop_masks(maskarray, scale=scale)
        # print('mask.shape', im.shape)
        yield get_square(im, pos)

def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.png', scale)


    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_masks(ids, dir_mask, '_mask.npz', scale)

    return zip(imgs_normalized, masks)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '_mask.gif')
    return np.array(im), np.array(mask)


class Dataset_predict(data.Dataset):
    'Characterizes a dataset for PyTorch'
    'input paths of img and labels, output with img size (H, W, C), label (H ,W, C)'
    'input imgname :   00034.png'
    'input labelname:  00034_mask.npz'

    def __init__(self, imgpath, inputtype="L", scale=0.5):
        'Initialization'

        self.imgpath = imgpath
        self.imgids = [f for f in os.listdir(imgpath)]
        self.inputtype = inputtype
        self.scale = scale

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.imgids)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        imgid = self.imgids[index]
        img = Image.open(self.imgpath + '/' + imgid).convert(self.inputtype)
        w, h = img.size
        neww, newh = int(w*self.scale), int(h*self.scale)
        img = img.resize((neww, newh), Image.ANTIALIAS)
        img = np.array(img)
        if len(img.shape) == 2:
            img = img[..., np.newaxis]
        img = normalize(img)
        return img, imgid

