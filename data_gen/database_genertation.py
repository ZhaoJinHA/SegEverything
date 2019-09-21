<<<<<<< HEAD
##  generate a image database from only one image and its label

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from PIL import Image
from skimage.color import rgb2hsv
from utils import img_distortion
from img_augmentation import object_transformation

def image_inpainting(img_hole, img_mask):
    dst = cv2.inpaint(img_hole,img_mask,3,cv2.INPAINT_TELEA)
    return dst

def database_gen():
        
    imgsavedir = 'D:/Experiment_data/Tacoma Bridge/segdata/train/img'
    imgmasksavedir = 'D:/Experiment_data/Tacoma Bridge/segdata/train/mask'

    inputgray = 1

    imgsaveformat = 'jpg'     # input gray image. "RGB"
    imgreadmode = 'rgb'
    if inputgray:
        imgsaveformat = 'png'
        imgreadmode = 'L'

    imgroot = 'Y:\script\imagetest\image'.replace('\\','/')
    img = Image.open(os.path.join(imgroot,"frame3.png")).convert(imgreadmode)
    img = np.array(img)
    img_mask = np.load(os.path.join(imgroot,"ml_lbl2.npz"))['arr_0']
    datanum = 1000
    for i in range(datanum):
        imgout, img_maskout, auglist = data_gen(img,img_mask)
        auglist = "".join(map(str,auglist.astype(int)))
        
        # print(f'{i:05}.{imgsaveformat}')
        # print(os.path.join(imgsavedir, f'{i:05}.{imgsaveformat}'))
        if inputgray:
            imgout = imgout[:,:,0]
            Image.fromarray(imgout).convert('L').save(f'{imgsavedir}/{i:05}_{auglist}.{imgsaveformat}')   # auglist 101  simard,lumination,noise
        
        if not inputgray:
            Image.fromarray(imgout).convert('rgb').save(f'{imgsavedir}/{i:05}_{auglist}.{imgsaveformat}')
        
        np.savez_compressed(f'{imgmasksavedir}/{i:05}_mask_{auglist}.npz', label=img_maskout)  ## default name is arr_0

    return imgout, img_maskout
def data_gen(img, img_mask):
    objtf = object_transformation(img, img_mask)
    
    randomlist = np.random.rand(4)
    randomlist[randomlist>0.5] = 1
    randomlist[randomlist<=0.5] = 0
    
    while np.max(randomlist) == 0:
        randomlist = np.random.rand(4)
        randomlist[randomlist>0.5] = 1
        randomlist[randomlist<=0.5] = 0
        
    if randomlist[0]:
        print('simard dist')
        objtf.random_distor_simard()
    
    lumi = 0
    if randomlist[1]:
        print('lumination')
        lumi = np.random.uniform(-3,3)
        
    if randomlist[2]:
        print('noise')
        objtf.lumination_and_noise(lumi)

    if randomlist[3]:
        print('flip')
        objtf.random_flip()  
     

    imgout = objtf.obj
    img_maskout = objtf.obj_mask
    return imgout, img_maskout, randomlist


def main():
    pass
if __name__ == '__main__':
    database_gen()
=======
##  generate a image database from only one image and its label

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from PIL import Image
from skimage.color import rgb2hsv
from utils import img_distortion
from img_augmentation import object_transformation

def image_inpainting(img_hole, img_mask):
    dst = cv2.inpaint(img_hole,img_mask,3,cv2.INPAINT_TELEA)
    return dst

def database_gen():
        
    imgsavedir = 'D:/Experiment_data/Tacoma Bridge/segdata/train/img'
    imgmasksavedir = 'D:/Experiment_data/Tacoma Bridge/segdata/train/mask'

    inputgray = 1

    imgsaveformat = 'jpg'     # input gray image. "RGB"
    imgreadmode = 'rgb'
    if inputgray:
        imgsaveformat = 'png'
        imgreadmode = 'L'

    imgroot = 'Y:\script\imagetest\image'.replace('\\','/')
    img = Image.open(os.path.join(imgroot,"frame3.png")).convert(imgreadmode)
    img = np.array(img)
    img_mask = np.load(os.path.join(imgroot,"ml_lbl2.npz"))['arr_0']
    datanum = 1000
    for i in range(datanum):
        imgout, img_maskout, auglist = data_gen(img,img_mask)
        auglist = "".join(map(str,auglist.astype(int)))
        
        # print(f'{i:05}.{imgsaveformat}')
        # print(os.path.join(imgsavedir, f'{i:05}.{imgsaveformat}'))
        if inputgray:
            imgout = imgout[:,:,0]
            Image.fromarray(imgout).convert('L').save(f'{imgsavedir}/{i:05}_{auglist}.{imgsaveformat}')   # auglist 101  simard,lumination,noise
        
        if not inputgray:
            Image.fromarray(imgout).convert('rgb').save(f'{imgsavedir}/{i:05}_{auglist}.{imgsaveformat}')
        
        np.savez_compressed(f'{imgmasksavedir}/{i:05}_mask_{auglist}.npz', label=img_maskout)  ## default name is arr_0

    return imgout, img_maskout
def data_gen(img, img_mask):
    objtf = object_transformation(img, img_mask)
    
    randomlist = np.random.rand(4)
    randomlist[randomlist>0.5] = 1
    randomlist[randomlist<=0.5] = 0
    
    while np.max(randomlist) == 0:
        randomlist = np.random.rand(4)
        randomlist[randomlist>0.5] = 1
        randomlist[randomlist<=0.5] = 0
        
    if randomlist[0]:
        print('simard dist')
        objtf.random_distor_simard()
    
    lumi = 0
    if randomlist[1]:
        print('lumination')
        lumi = np.random.uniform(-3,3)
        
    if randomlist[2]:
        print('noise')
        objtf.lumination_and_noise(lumi)

    if randomlist[3]:
        print('flip')
        objtf.random_flip()  
     

    imgout = objtf.obj
    img_maskout = objtf.obj_mask
    return imgout, img_maskout, randomlist


def main():
    pass
if __name__ == '__main__':
    database_gen()
>>>>>>> first commit
