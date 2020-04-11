import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from PIL import Image
from skimage.color import rgb2hsv
import argparse

def hsv2color(hsv):
    h, s, v = hsv[0], hsv[1], hsv[2]
    if v < 0.07:
        return 'k'
    if 0<=h<=15 or 350<=h<=360:
        return 'r'
    if 100<=h<=140:
        return 'g'
        

def hsv2color_map(hsv, numcls):
    hsvshape = hsv.shape
    color_map=np.zeros((hsvshape[0], hsvshape[1]))    # 0 represents the black background
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    h = h*360
    color_map[np.where((((0<=h) & (h<=15)) | ((350<=h) & (h<=360))) & (s>=0.5) )] = 1
    color_map[np.where(((100<=h) & (h<=140))& (s>=0.5))] = 2
    color_map[np.where(((45<=h) & (h<=60))& (s>=0.5))] = 3
    color_map[np.where(((200 <= h) & (h <= 280)) & (s >= 0.5))] = 4
    print(h)
    print(color_map)
    
    color_layers = np.zeros((hsvshape[0], hsvshape[1], numcls))
    for i in range(numcls):
        color_layers[:, :, i][np.where(color_map==i)] = 1
        # print(np.where(color_map==i).shape)
        
    return color_layers
    
def rgb2multilayers(img, numcls, clrlist=['k', 'g', 'r', 'o']):
    img_hsv = rgb2hsv(img)
       
    return hsv2color_map(img_hsv, numcls)

def multilayers_save(labels, path):
    np.savez_compressed(path, data=labels)

def get_args(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgbpicfile', '-r', default='/home/zhaojin/NutstoreFiles/Nutstore/maanshan/马鞍山大桥抖振资料/label_result2/label.png',
                        help="Specify the rgblabel file path"
                             " (default : 'MODEL.pth')")

    parser.add_argument('--output', '-o', default='/home/zhaojin/data/maanshan/segdata/maanshan.npz',
                        help='filenames of ouput images')

    return parser.parse_args(raw_args)

def rgblbl2npz(raw_args=None):
    args = get_args(raw_args)
    rgbpicfile = args.rgbpicfile
    print('rgbpicfile', rgbpicfile)
    npzfile = args.output
    img = Image.open(rgbpicfile).convert("RGB")

    numcls = 3
    lbl_lay = rgb2multilayers(img, numcls)
    lay0 = lbl_lay[..., 2]
    print('lay0.shape', lay0.shape)
    print('np.max(lay0)', np.max(lay0))
    print('np.sum(lay0)', np.sum(lay0))


    multilayers_save(lbl_lay, npzfile)

    lbl_lay = np.array(lbl_lay * 255, dtype="uint8")

    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    ax.imshow(img)


    fig = plt.figure()
    ax = fig.add_subplot(141)
    ax.imshow(lbl_lay[:, :, 0], cmap='gray')
    ax.set_aspect(1)


    ax1 = fig.add_subplot(142)
    ax1.imshow(lbl_lay[:, :, 1])
    ax1.set_aspect(1)

    ax1 = fig.add_subplot(143)
    ax1.imshow(lbl_lay[:, :, 2])
    ax1.set_aspect(1)


    ax1.set_aspect(1)


    # ax2 = fig.add_subplot(143)
    # ax2.imshow(lbl_lay[:, :, 3])
    # ax2.set_aspect(1)
    #
    # ax3 = fig.add_subplot(144)
    # ax3.imshow(lbl_lay[:, :, 4])
    # ax3.set_aspect(1)





    plt.show()

if __name__ == "__main__":
    rgblbl2npz()

