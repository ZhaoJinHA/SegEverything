# SegEverything (what a big name)
A simple tool to help to train a semantic segmentation network for a video or a set of images.



## Workflow
- label the first frame of the video use labelme [https://github.com/wkentaro/labelme]
- generate the database for trainning
- train a U-net-like network



## Independence

- now work in windows 10. However, it is easy to modified to Ubuntu

- python >= 3.6
- installing [labelme](<https://gist.github.com/erniejunior/601cdf56d2b424757de5>)

- pytorch with version >= 0.4.1

- skimage, PIL, [pydensecrf](https://github.com/lucasb-eyer/pydensecrf) (installed with pip)

  

## Reference website

- the semantic segmentation code is modified from [pytorch-unet](https://github.com/milesial/Pytorch-UNet)
- img_distortion part of database generation code is modified from [https://gist.github.com/erniejunior/601cdf56d2b424757de5]



## Progress

- [x] convert the label from labelme format to numpy.npz format
- [x] complete the data generation code
- [x] change the loss function in Pytorch-unet project from log loss function to weight loss function, and other modification
- [ ] make all the workflow more auto
- [ ] make it work in Ubuntu
