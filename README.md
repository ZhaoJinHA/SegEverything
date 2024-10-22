# SegEverything (what a big name)
A simple tool to help to train a semantic segmentation network for a video or a set of images.



## Workflow
- label the first frame of the video use labelme [https://github.com/wkentaro/labelme]
- generate the database for trainning
- train a U-net-like network


## Hardware
- Nvidia Graphic cards for cuda trainning.

## Independence

- now work in windows 10 ~~. However, it is easy to modified to~~ and Ubuntu

- python >= 3.6
- ~~installing [labelme](<https://gist.github.com/erniejunior/601cdf56d2b424757de5>)~~

- pytorch with version >= 0.4.1

- skimage, PIL, [pydensecrf](https://github.com/lucasb-eyer/pydensecrf) (installed with pip)

## Usage
- Firstly, you should have your first frame of video labeled. This step could be done by anyway you want. The label result should be stored in a npz file. The data formation should be a numpy array with size (W, H, C) (weight, height of the image, and channels for your labels), and this label should only contain 0 and 1. If you don't know how to generate this type of file, you could use ./data_gen/utils/rgb2multilayers to help you convert a colorful label (Figure 1) into a multilayer formation label. The color list is :[black, green, red, orange] for rightnow, which means that you can do classification of equal or less than 4. 

<p align="center">
  <img width="640" height="480" src="https://github.com/ZhaoJinHA/SegEverything/blob/master/label_example.png">
</p>

<p align="center">
  Figure 1: rgb label example
</p>



- To predict a picture: example

 python predict.py --model 'path/to/model.pth' --input 'path/to/image/to/predict' --viz

- To predict pictures from a video (size same size)

 python predict_batch.py --model 'path/to/model' --input 'path/to/images/' --output 'path/to/path/to/save/result'

- To train a net: example

1. python train.py -i 'path/to/image/' -m 'path/to/masks' -v 'checkpointsavepath' -l 0.1 -d 0.99 -e 30 -b 10

2. python train.py -i 'path/to/image/' -m 'path/to/masks' -v 'path/to/save/checkpoin'



## Reference website

- the semantic segmentation code is modified from [pytorch-unet](https://github.com/milesial/Pytorch-UNet)

- img_distortion part of database generation code is modified from [https://gist.github.com/erniejunior/601cdf56d2b424757de5]



## Progress

- [x] convert the label from labelme format to numpy.npz format
- [x] complete the data generation code
- [x] change the loss function in Pytorch-unet project from log loss function to weight loss function, and other modification
- [x] make all the workflow more auto, use ./train_and_predict.py and ./data_gen/database_test.py
- [x] make it work in Ubuntu



