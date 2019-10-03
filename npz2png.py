
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

if __name__ == '__main__':
    npz2png()