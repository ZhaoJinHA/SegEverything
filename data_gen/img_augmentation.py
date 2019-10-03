## operate image augmetation method to a image and its correspoding label

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from utils import img_distortion
from scipy.ndimage.filters import gaussian_filter


class object_transformation():
    """control the transofrmation of the object in a image """
    def __init__ (self, obj, obj_mask):
        self.obj0 = obj
        self.obj_mask0 = obj_mask
        self.obj = obj
        self.obj_mask = obj_mask
        self.obj_shape = obj.shape
        self.obj_size = self.obj_shape[:2]
        self.ifgray = len(self.obj_shape) == 2
        
        if self.ifgray:
            self.obj = self.obj[...,np.newaxis]
    
    def sinlike_tf(self, direction):
        """ direction denotes the transformation direction degree."""
        if direction==90:  #in 90 degree direction (y direction)
            imgy, imgx= np.where(img>=200)
            x1 = np.min(imgx)
            x2 = np.max(imgx)
            y1= np.min(imgy)
            y2 = np.max(imgy)
            objlen = x2 - x1
        pass
    
    def random_distor_simard(self):
        obj, obj_mask = img_distortion(self.obj, self.obj_mask)
        self.obj = obj
        self.obj_mask = obj_mask
        print('method simard applied!')
        
    def lumination_and_noise(self, lumi_val=0, noise_val=3.5, sigma=1):
        noise = np.random.rand(*self.obj_size)*(2*noise_val) - (noise_val - lumi_val)
        noise_gau = gaussian_filter(noise, sigma)
        if len(self.obj_shape) == 2:
            self.obj = self.obj + noise_gau[...,np.newaxis]
            self.obj[self.obj<0] = 0
            self.obj[self.obj>255] = 255
        if len(self.obj_shape) == 3:
            objlab = rgb2lab(self.obj)
            objgray = objlab[...,0] + noise_gau
            objgray[objgray<0] = 0
            objgray[objgray>255] = 255
            objlab[...,0] = objgray
            self.obj = (lab2rgb(objlab)*255).astype('uint8')
        
    def camera_rotate_and_translation(self):
        pass
    
    def random_flip(self):
        self.obj = np.fliplr(self.obj)
        self.obj_mask = np.fliplr(self.obj_mask)
        
    def objout(self):
        return self.obj

def main():

    img = Image.open('/home/zhaojin/data/TacomaBridge/segdata/train/img/00000.png').convert("L")
    img = np.array(img)
    img_mask = np.load('/home/zhaojin/data/TacomaBridge/segdata/train/mask/00000_mask.npz')['label']

    # print('img_mask.shape',img_mask.shape)
    # print('np.max(img_mask)',np.max(img_mask))

    # print('img.shape',img.shape)
    # print('np.max(img)',np.max(img))
    objtf = object_transformation(img, img_mask)
    objtf.random_distor_simard()
    lumi = np.random.uniform(-3,3)
    objtf.lumination_and_noise(lumi)

    imgout = objtf.obj
    img_maskout = objtf.obj_mask

    print('imgout.shape',imgout.shape)
    print('img_maskout.shape',img_maskout.shape)
    print('np.max(imgout)',np.max(imgout))
    print('np.max(img_maskout)',np.max(img_maskout))

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(img_maskout[...,0])
    ax.set_aspect(1)

    ax = fig.add_subplot(122)
    ax.imshow(imgout[...,0])
    ax.set_aspect(1)


    plt.show()

if __name__ == '__main__':
    main()