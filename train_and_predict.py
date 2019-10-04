import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import predict
from train import train_1st
from predict_batch import main
from npz2png import npz2png
def safecreate(path):
    if not os.path.exists(path):
        os.mkdir(path)

def main1():
    trainimgpath='/home/zhaojin/data/TacomaBridge/segdata/train/img/'
    checkpointsavepath='/home/zhaojin/data/TacomaBridge/segdata/train/checkpoint/'
    checkpointreadpath='/home/zhaojin/data/TacomaBridge/segdata/train/checkpoint/weight_logloss_softmax/CP30.pth'

    predictimgname='/home/zhaojin/data/TacomaBridge/segdata/train/img/00034.png'
    predictimgpath='/home/zhaojin/data/TacomaBridge/capture/high-reso-clip2_rename/'
    predictlblpath='/home/zhaojin/data/TacomaBridge/segdata/predict/'
    predictlblvizpath=predictlblpath.rstrip('/') + "_viz"

    predict.main(['--model',checkpointreadpath, '--input',predictimgname, '--viz'])


def main2():
    trainimgpath = '/home/zhaojin/data/TacomaBridge/segdata/train2/img/'
    trainlblpath = '/home/zhaojin/data/TacomaBridge/segdata/train2/mask/'
    checkpointsavepath = '/home/zhaojin/data/TacomaBridge/segdata/train/checkpoint/weight_logloss_softmax2/'
    checkpointreadpath = '/home/zhaojin/data/TacomaBridge/segdata/train/checkpoint/weight_logloss_softmax2/CP30.pth'

    predictimgname = '/home/zhaojin/data/TacomaBridge/segdata/train/img/00034.png'
    predictimgpath = '/home/zhaojin/data/TacomaBridge/capture/high-reso-clip2_rename/'
    predictlblpath = '/home/zhaojin/data/TacomaBridge/segdata/predict'
    predictlblvizpath = predictlblpath.rstrip('/') + "_viz"

    safecreate(checkpointsavepath)

    epoch = 30
    lr = 0.05
    lrd = 0.87
    batch = 10


    codeinput = ['-i', trainimgpath, '-m', trainlblpath, '-v', checkpointsavepath, '-e', str(epoch), '-l', str(lr), '-d', str(lrd), '-b', str(batch) ]
    train_1st(codeinput)
    # predict.main(['--model', checkpointreadpath, '--input', predictimgname, '--viz'])

def main3():
    trainimgpath = '/home/zhaojin/data/TacomaBridge/segdata/train4&5/img/'
    trainlblpath = '/home/zhaojin/data/TacomaBridge/segdata/train4&5/mask/'
    checkpointsavepath = '/home/zhaojin/data/TacomaBridge/segdata/train/checkpoint/weight_logloss_softmax4&5/'
    checkpointreadpath = '/home/zhaojin/data/TacomaBridge/segdata/train/checkpoint/weight_logloss_softmax4&5/CP30.pth'

    predictimgname = '/home/zhaojin/data/TacomaBridge/segdata/train/img/00034.png'
    predictimgpath = '/home/zhaojin/data/TacomaBridge/capture/high-reso-clip2_rename/'
    predictlblpath = '/home/zhaojin/data/TacomaBridge/segdata/predict'
    predictlblvizpath = predictlblpath.rstrip('/') + "_viz"

    safecreate(checkpointsavepath)

    epoch = 30
    lr = 0.05
    lrd = 0.87
    batch = 10


    codeinput = ['-i', trainimgpath, '-m', trainlblpath, '-v', checkpointsavepath, '-e', str(epoch), '-l', str(lr), '-d', str(lrd), '-b', str(batch) ]
    train_1st(codeinput)
    # predict.main(['--model', checkpointreadpath, '--input', predictimgname, '--viz'])

def batch_predict():
    print('start batch precit')
    trainimgpath = '/home/zhaojin/data/TacomaBridge/segdata/train2/img'
    trainlblpath = '/home/zhaojin/data/TacomaBridge/segdata/train2/mask'
    checkpointsavepath = '/home/zhaojin/data/TacomaBridge/segdata/train/checkpoint/weight_logloss_softmax2'
    checkpointreadpath = '/home/zhaojin/data/TacomaBridge/segdata/train/checkpoint/weight_logloss_softmax2/CP30.pth'

    predictimgname = '/home/zhaojin/data/TacomaBridge/segdata/train/img/00034.png'
    predictimgpath = '/home/zhaojin/data/TacomaBridge/capture/high-reso-clip2_rename/'
    predictlblpath = '/home/zhaojin/data/TacomaBridge/segdata/predict2/high-reso-clip2-reanme/'

    predictlblvizpath = predictlblpath.rstrip('/') + "_viz"

    safecreate(predictlblpath)
    safecreate(predictlblvizpath)

    inputcode = ['--model',checkpointreadpath ,'--input', predictimgpath, '--output', predictlblpath]

    main(inputcode)

    npz2png(lblpath=predictlblpath, imgoutpath=predictlblvizpath)

def batch_predict2():
    print('start batch precit')
    trainimgpath = '/home/zhaojin/data/TacomaBridge/segdata/train2/img'
    trainlblpath = '/home/zhaojin/data/TacomaBridge/segdata/train2/mask'
    checkpointsavepath = '/home/zhaojin/data/TacomaBridge/segdata/train/checkpoint/weight_logloss_softmax2'
    checkpointreadpath = '/home/zhaojin/data/TacomaBridge/segdata/train4&5/weight_logloss_softmax4&5/CP30.pth'

    predictimgname = '/home/zhaojin/data/TacomaBridge/segdata/train/img/00034.png'
    predictimgpath = '/home/zhaojin/data/TacomaBridge/capture/high-reso-clip2_rename/'
    predictlblpath = '/home/zhaojin/data/TacomaBridge/segdata/train4&5/predict/high-reso-clip2-reanme/'

    predictlblvizpath = predictlblpath.rstrip('/') + "_viz"

    safecreate(predictlblpath)
    safecreate(predictlblvizpath)

    inputcode = ['--model', checkpointreadpath, '--input', predictimgpath, '--output', predictlblpath]

    main(inputcode)

    npz2png(lblpath=predictlblpath, imgoutpath=predictlblvizpath)
if __name__ == "__main__":
    # batch_predict()
    # main3()
    batch_predict2()