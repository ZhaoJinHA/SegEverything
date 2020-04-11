
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def npz2png(lblpath=None, imgoutpath=None):
    if lblpath == None:
        lblpath = '/home/zhaojin/data/TacomaBridge/segdata/predict/high-reso-clip2_rename_noweight'

    if imgoutpath==None:
        imgoutpath = lblpath.rstrip('/') + '_viz/'
    namelist = [f for f in os.listdir(lblpath)]
    for n, name in enumerate(namelist):
        npzfilename = lblpath + '/' + name
        npzdata= np.load(npzfilename)
        npzkey = [n for n in npzdata][0]
        nparr = npzdata[npzkey]
        img = np.array(nparr*255,dtype='uint8')
        Image.fromarray(img).convert("L").save(imgoutpath + name.split('.npz')[0] + '.png')

def npz2png_gray(lblpath=None, imgoutpath=None):
    if lblpath == None:
        lblpath = '/home/zhaojin/data/TacomaBridge/segdata/predict/high-reso-clip2_rename_noweight'

    if imgoutpath==None:
        imgoutpath = lblpath.rstrip('/') + '_viz/'
    namelist = [f for f in os.listdir(lblpath)]
    for n, name in enumerate(namelist):
        npzfilename = lblpath + '/' + name
        npzdata= np.load(npzfilename)
        npzkey = [n for n in npzdata][0]
        nparr = npzdata[npzkey]
        imgshape = nparr.shape
        # print('imgshape', imgshape )
        imgout = np.zeros((imgshape[0], imgshape[1]))
        # print('imgout.shape', imgout.shape )
        graylist = np.linspace(0,1, imgshape[2])
        for ic in range(imgshape[2]):
            imgc = nparr[..., ic]
            # print('np.sum(imgc)', np.sum(imgc) )
            # print('imgc', imgc )
            imgout[np.where(imgc > 0.5)] = graylist[ic]



        imgout = np.array(imgout*255,dtype='uint8')
        # print('np.max(imgout)', np.max(imgout) )
        # print('np.min(imgout)', np.min(imgout) )
        Image.fromarray(imgout).convert("L").save(imgoutpath + name.split('.npz')[0] + '.png')

if __name__ == '__main__':
    npz2png()