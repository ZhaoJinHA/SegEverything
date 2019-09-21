<<<<<<< HEAD
# Import stuff
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import os 
# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    # print('shape',shape)
    
    # print('shape_size',shape_size)
    # print('shape_size[::-1]',shape_size[::-1])
    
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    # print('pts1',pts1)
    # print('pts2',pts2)
    
    M = cv2.getAffineTransform(pts1, pts2)
    if 1:
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    # print('indices',indices)
    # print('image',image.shape)
    
    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    
# Define function to draw a grid
def draw_grid(im, grid_size):
    # Draw grid lines
    im = im.astype('uint8')
    if np.max(im) <= 2.0:
        im = im*255
    for i in range(0, im.shape[1], grid_size):
        
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(0,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(0,))

def img_distortion(im, im_mask):
    # img : (0,255)  imgmask(0,1)
    imshape = im.shape
    im_maskshape = im_mask.shape
    if len(imshape) == 2:
        im = im[...,np.newaxis]
        imshape = im.shape
    if len(im_maskshape) == 2:
        im_mask = im_mask[...,np.newaxis]
        im_maskshape = im_mask.shape
    im_merge = np.concatenate((im, im_mask), axis=2)
    im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)
    imgout = im_merge_t[...,:imshape[2]].astype('uint8')
    maskout = im_merge_t[...,imshape[2]:]
    
    
    return imgout, maskout

def main():
    imgroot = 'Y:\script\imagetest\image'
    im = cv2.imread(os.path.join(imgroot,'frame3.png'), -1)
    # print('im',im.shape)

    loadeddata = np.load(os.path.join(imgroot,'ml_lbl.npz'))
    im_mask = loadeddata[loadeddata.files[0]]

    # Draw grid lines
    # draw_grid(im, 50)
    # draw_grid(im_mask[...,1], 50)

    # Merge images into separete channels (shape will be (cols, rols, 2))
    im_merge = np.concatenate((im, im_mask), axis=2)
    # First sample...


    # Apply transformation on image
    im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)

    # Split image and mask
    im_t = im_merge_t[...,0]
    im_mask_t = im_merge_t[...,4]

    # Display result
    plt.figure(figsize = (16,14))
    plt.imshow(np.c_[np.r_[im[...,1], im_mask[...,1]*255], np.r_[im_t, im_mask_t*255]],cmap='gray')
    # plt.imshow(im_mask[...,1]*255)

    plt.show()
    
def main2():
    imgroot = 'Y:\script\imagetest\image'
    im = cv2.imread(os.path.join(imgroot,'frame3.png'), -1)
    # print('im',im.shape)

    loadeddata = np.load(os.path.join(imgroot,'ml_lbl.npz'))
    im_mask = loadeddata[loadeddata.files[0]]
    
    imout, labelout = img_distortion(im, im_mask)
    
    # print('imout',imout)
    
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.imshow(imout)
    ax.set_aspect(1)
    
    ax = fig.add_subplot(132)
    ax.imshow(im_mask[...,0])
    ax.set_aspect(1)
    
    
    ax = fig.add_subplot(133)
    ax.imshow(labelout[...,0],cmap='gray')
    ax.set_aspect(1)
    plt.show()
    
    return imout, labelout
    
   
    
if __name__ == '__main__':
    print('img_distortion, main2()')
    imout, labelout = main2()
# Load images

=======
# Import stuff
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import os 
# Function to distort image
def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    # print('shape',shape)
    
    # print('shape_size',shape_size)
    # print('shape_size[::-1]',shape_size[::-1])
    
    
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    # print('pts1',pts1)
    # print('pts2',pts2)
    
    M = cv2.getAffineTransform(pts1, pts2)
    if 1:
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    # print('indices',indices)
    # print('image',image.shape)
    
    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    
# Define function to draw a grid
def draw_grid(im, grid_size):
    # Draw grid lines
    im = im.astype('uint8')
    if np.max(im) <= 2.0:
        im = im*255
    for i in range(0, im.shape[1], grid_size):
        
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(0,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(0,))

def img_distortion(im, im_mask):
    # img : (0,255)  imgmask(0,1)
    imshape = im.shape
    im_maskshape = im_mask.shape
    if len(imshape) == 2:
        im = im[...,np.newaxis]
        imshape = im.shape
    if len(im_maskshape) == 2:
        im_mask = im_mask[...,np.newaxis]
        im_maskshape = im_mask.shape
    im_merge = np.concatenate((im, im_mask), axis=2)
    im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)
    imgout = im_merge_t[...,:imshape[2]].astype('uint8')
    maskout = im_merge_t[...,imshape[2]:]
    
    
    return imgout, maskout

def main():
    imgroot = 'Y:\script\imagetest\image'
    im = cv2.imread(os.path.join(imgroot,'frame3.png'), -1)
    # print('im',im.shape)

    loadeddata = np.load(os.path.join(imgroot,'ml_lbl.npz'))
    im_mask = loadeddata[loadeddata.files[0]]

    # Draw grid lines
    # draw_grid(im, 50)
    # draw_grid(im_mask[...,1], 50)

    # Merge images into separete channels (shape will be (cols, rols, 2))
    im_merge = np.concatenate((im, im_mask), axis=2)
    # First sample...


    # Apply transformation on image
    im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)

    # Split image and mask
    im_t = im_merge_t[...,0]
    im_mask_t = im_merge_t[...,4]

    # Display result
    plt.figure(figsize = (16,14))
    plt.imshow(np.c_[np.r_[im[...,1], im_mask[...,1]*255], np.r_[im_t, im_mask_t*255]],cmap='gray')
    # plt.imshow(im_mask[...,1]*255)

    plt.show()
    
def main2():
    imgroot = 'Y:\script\imagetest\image'
    im = cv2.imread(os.path.join(imgroot,'frame3.png'), -1)
    # print('im',im.shape)

    loadeddata = np.load(os.path.join(imgroot,'ml_lbl.npz'))
    im_mask = loadeddata[loadeddata.files[0]]
    
    imout, labelout = img_distortion(im, im_mask)
    
    # print('imout',imout)
    
    fig = plt.figure()
    ax = fig.add_subplot(131)
    ax.imshow(imout)
    ax.set_aspect(1)
    
    ax = fig.add_subplot(132)
    ax.imshow(im_mask[...,0])
    ax.set_aspect(1)
    
    
    ax = fig.add_subplot(133)
    ax.imshow(labelout[...,0],cmap='gray')
    ax.set_aspect(1)
    plt.show()
    
    return imout, labelout
    
   
    
if __name__ == '__main__':
    print('img_distortion, main2()')
    imout, labelout = main2()
# Load images

>>>>>>> first commit
