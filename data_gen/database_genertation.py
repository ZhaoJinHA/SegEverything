##  generate a image database from only one image and its label
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from PIL import Image
from skimage.color import rgb2hsv
# from .utils import img_distortion
from img_augmentation import object_transformation
import cv2

def image_inpainting(img_hole, img_mask):
    dst = cv2.inpaint(img_hole,img_mask,3,cv2.INPAINT_TELEA)
    return dst

def database_gen(imgpath, labelpath, inputgray=True,  outpath='D:/Experiment_data/Tacoma Bridge/segdata/train/', lumi=10, noise=5, datanum=1000):

    imgsavedir = outpath + '/img'
    imgmasksavedir = outpath + '/mask'

    imgsaveformat = 'jpg'     # input gray image. "RGB"
    imgreadmode = 'rgb'
    if inputgray:
        imgsaveformat = 'png'
        imgreadmode = 'L'

    img = Image.open(imgpath).convert(imgreadmode)
    img = np.array(img)

    img_mask = np.load(labelpath)
    keyname = [item for item in img_mask]
    img_mask = img_mask[keyname[0]]

    for i in range(datanum):

        print('datanum', i )
        imgout, img_maskout, auglist = data_gen(img,img_mask, lumi, noise)
        auglist = "".join(map(str,auglist.astype(int)))
        
        # print(f'{i:05}.{imgsaveformat}')
        # print(os.path.join(imgsavedir, f'{i:05}.{imgsaveformat}'))
        if inputgray:
            imgout = imgout[:,:,0]
            Image.fromarray(imgout).convert('L').save(f'{imgsavedir}/{i:05}.{imgsaveformat}')   # auglist 101  simard,lumination,noise
        
        if not inputgray:
            Image.fromarray(imgout).convert('rgb').save(f'{imgsavedir}/{i:05}.{imgsaveformat}')
        
        np.savez_compressed(f'{imgmasksavedir}/{i:05}_mask.npz', label=img_maskout)  ## default name is arr_0

    return imgout, img_maskout


def data_gen(img, img_mask, lumi_val=5, noise_val=3.5):
    objtf = object_transformation(img, img_mask)
    
    randomlist = np.random.rand(3)
    transform_pro = 0.2                     # transform_pro determine the probability of every transfromation. If the value is 0.5, the transformation probability is too low.
    randomlist[randomlist > transform_pro] = 1
    randomlist[randomlist <= transform_pro] = 0
    
    while np.max(randomlist) == 0:
        randomlist = np.random.rand(3)
        randomlist[randomlist > transform_pro] = 1
        randomlist[randomlist <= transform_pro] = 0
        
    if randomlist[0]:
        # print('simard dist')
        objtf.random_distor_simard()
    
    lumi = 0
    if randomlist[1]:
        # print('lumination')
        # print('type(lumi_val)', type(lumi_val) )
        lumi = np.random.uniform(-lumi_val,lumi_val)

    noise=0
    if randomlist[2]:
        # print('noise')
        noise = np.random.uniform(noise_val)
    objtf.lumination_and_noise(lumi, noise)

    if np.random.rand(1) > 0.5:                    # random flip probability should be 0.5.
        # print('flip')
        objtf.random_flip()  
     

    imgout = objtf.obj
    img_maskout = objtf.obj_mask

    return imgout, img_maskout, randomlist

def get_args(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lumination', '-l', default=10, type=int,
                        help="Specify the lumination change mean value"
                             " (default : 10")
    parser.add_argument('--noise', '-n', default=5, type=int,
                        help="Specify the noise change var"
                             " (default : 5")
    parser.add_argument('--img', '-i', help='filenames of input image')
    parser.add_argument('--label', '-b', help='filenames of input label')
    parser.add_argument('--output', '-o', help='output path of ouput images')
    parser.add_argument('--datanum', '-d', help='number of dataset images', type=int)


    return parser.parse_args(raw_args)

def main(raw_args=None):
    args = get_args(raw_args)
    lumi = args.lumination
    noise = args.noise
    outpath = args.output
    imgpath = args.img
    labelpath = args.label
    datanum=args.datanum
    database_gen(imgpath, labelpath, True,  outpath, lumi, noise,datanum=datanum)
if __name__ == '__main__':
    main()
