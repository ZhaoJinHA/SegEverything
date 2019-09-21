# SegEverything (what a big name)
A simple tool to help to train a semantic segmentation network for a video or a set of images.

## workflow
- label the first frame of the video use labelme [https://github.com/wkentaro/labelme]
- generate the database for trainning
- train 
##  
the semantic segmentation code is from pytorch-unet [https://github.com/milesial/Pytorch-UNet]
img_distortion part of database generation code is from [https://gist.github.com/erniejunior/601cdf56d2b424757de5]