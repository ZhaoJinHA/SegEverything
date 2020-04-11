import random
import numpy as np
from PIL import Image


def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (H, W, C))"""
    h = img.shape[0]
    if pos == 0:
        return img[:, :h]
    else:
        return img[:, -h:]

def split_img_into_squares(img):
    """ img with size: (H, W, C) """
    return get_square(img, 0), get_square(img, 1)

def split_img_into_squares_batch(img):
    """ img with size: (N, H, W, C) """
    h = img.shape[1]
    return img[:, :, :h, :], img[:, :, -h:, :]

def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])

def resize_and_crop(pilimg, scale=1, final_height=None):
    """ imput PIL format image, output with np.array np.floa32 type, with max 255 """
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    if not scale == 1:
        img = pilimg.resize((newW, newH))
    else:
        img = pilimg
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return np.array(img, dtype=np.float32)

def resize_and_crop_masks(masksarray, scale=1, final_height=None):
    """ input with nparray with size (H, W, C) and range (0,1), output with np.array np.floa32 type, with max 1 """

    h = masksarray.shape[0]
    w = masksarray.shape[1]
    c = masksarray.shape[2]
    # print('masksarray.shape', masksarray.shape)
    newW = int(w * scale)
    newH = int(h * scale)

    maskout = np.zeros((newH, newW, c),dtype=np.float32)
    # print('maskout0.shape', maskout.shape)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height
    for iC in range(c):
        img = Image.fromarray(masksarray[...,iC])
        if not scale==1:
            img = img.resize((newW, newH))
        # print(img)
        img = img.crop((0, diff // 2, newW, newH - diff // 2))
        img = np.array(img, dtype=np.float32)
        maskout[...,iC] = img
    # print('maskout.shape', maskout.shape)
    return maskout

def resize_np(nparray, scale=0.5):
    """resize a (H, W, C) like numpy array, using PIL.reisze, return"""
    h = nparray.shape[0]
    w = nparray.shape[1]
    c = nparray.shape[2]
    # print('masksarray.shape', masksarray.shape)
    newW = int(w * scale)
    newH = int(h * scale)
    maskout = np.zeros((newH, newW, c))
    for iC in range(c):
        img = Image.fromarray(nparray[...,iC])
        if not scale==1:
            img = img.resize((newW, newH))

        img = np.array(img)
        maskout[...,iC] = img
    return maskout



def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b

def split_train_val_test(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    n=100
    random.shuffle(dataset)
    return {'train': dataset[:n], 'val': dataset[-n:]}

def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)

    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}

def normalize(x):
    return x / 255

def merge_masks(img1, img2, full_w):
    """img1 and img2 with size(C, H, W)"""
    h = img1.shape[1]
    c = img1.shape[0]

    new = np.zeros((c, h, full_w), np.float32)
    new[:, :, :full_w // 2 + 1] = img1[:, :, :full_w // 2 + 1]
    new[:, :, full_w // 2 + 1:] = img2[:, :, -(full_w // 2 - 1):]

    return new


def merge_masks_batch(img1, img2, full_w):
    """img1 and img2 with size(N, C, H, W)"""
    h = img1.shape[2]
    c = img1.shape[1]
    n = img1.shape[0]
    new = np.zeros((n, c, h, full_w), np.float32)
    new[:, :, :, :full_w // 2 + 1] = img1[:, :, :, :full_w // 2 + 1]
    new[:, :, :, full_w // 2 + 1:] = img2[:, :, :, -(full_w // 2 - 1):]

    return new

# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs
