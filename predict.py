## test for split
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from unet import UNet
from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks, dense_crf,resize_np
from utils import plot_img_and_mask

from torchvision import transforms

import matplotlib.pyplot as plt

def predict_img(net,
                full_img,
                scale_factor=0.5,
                out_threshold=0.5,
                use_dense_crf=True,
                use_gpu=True):
    """return fullmask with size (C, H, W)"""
    net.eval()
    img_height = full_img.size[1]
    img_width = full_img.size[0]
    print('imgheight', img_height)
    print('imgwidth', img_width)
    img = resize_and_crop(full_img, scale=scale_factor)
    img = normalize(img)
    
    if len(img.shape)==2:
        img = img[...,np.newaxis]

    print('img.shape', img.shape)
    left_square, right_square = split_img_into_squares(img)

    left_square = hwc_to_chw(left_square)
    right_square = hwc_to_chw(right_square)

    X_left = torch.from_numpy(left_square).unsqueeze(0)
    X_right = torch.from_numpy(right_square).unsqueeze(0)
    
    if use_gpu:
        X_left = X_left.cuda()
        X_right = X_right.cuda()

    with torch.no_grad():
        output_left = net(X_left)
        output_right = net(X_right)
        

        print('output_left.shape', output_left.shape)
        print('output_right.shape', output_right.shape)
        left_probs = output_left.squeeze(0)
        right_probs = output_right.squeeze(0)
        #
        # if not scale_factor==1:
        #     tf = transforms.Compose(
        #         [
        #             transforms.ToPILImage(),
        #             transforms.Resize(img_height),
        #             transforms.ToTensor()
        #         ]
        #     )
        #
        #     left_probs = tf(left_probs.cpu())
        #     right_probs = tf(right_probs.cpu())
        # print('left_probs', left_probs.shape)
        # print('right_probs', right_probs.shape)
        left_mask_np = left_probs.squeeze().cpu().numpy()
        right_mask_np = right_probs.squeeze().cpu().numpy()
        left_mask_np = np.transpose(left_mask_np, axes=[1,2,0])
        right_mask_np = np.transpose(right_mask_np, axes=[1,2,0])
        if not scale_factor == 1:
            right_mask_np = resize_np(right_mask_np, 1/scale_factor)
            left_mask_np = resize_np(left_mask_np, 1/scale_factor)
        print('left_mask_np', left_mask_np.shape )
        print('right_mask_np', right_mask_np.shape )
    left_mask_np = np.transpose(left_mask_np, axes=[2, 0, 1])
    right_mask_np = np.transpose(right_mask_np, axes=[2, 0, 1])
    full_mask = merge_masks(left_mask_np, right_mask_np, img_width)

    # if use_dense_crf:
    if 0:
        full_mask = dense_crf(np.array(full_img).astype(np.uint8), full_mask)

    return full_mask
    # return left_mask_np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--no-crf', '-r', action='store_true',
                        help="Do not use dense CRF postprocessing",
                        default=True)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()

def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))

def main():
    args = get_args()
    args.model = '/home/zhaojin/data/TacomaBridge/segdata/train/checkpoint/logloss_softmax/CP12.pth' 
    in_files = ['/home/zhaojin/data/TacomaBridge/segdata/train/img/00034.png' ]
    out_files = ['/home/zhaojin/my_path/dir/segdata/predict/00025.png']

    net = UNet(n_channels=1, n_classes=4)

    print("Loading model {}".format(args.model))

    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(torch.load(args.model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)
        if img.size[0] < img.size[1]:
            print("Error: image height larger than the width")

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           use_dense_crf= not args.no_crf,
                           use_gpu=not args.cpu)

        if args.viz:
            print("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
        # if not args.no_save:
        #     out_fn = out_files[i]
        #     print('mask', mask)
        #     result = mask_to_image(mask)
        #
        #     result.save(out_files[i])
        #
        #     print("Mask saved to {}".format(out_files[i]))

    return mask

if __name__ == "__main__":
    """example:  python predict.py --model '/home/zhaojin/data/TacomaBridge/segdata/train/checkpoint/weight_logloss_softmax/CP30.pth' --input '/home/zhaojin/data/TacomaBridge/segdata/train/img/00034.png' --viz"""
    args = get_args()
    # args.model = '/home/zhaojin/data/TacomaBridge/segdata/train/checkpoint/logloss_softmax/CP12.pth'
    # in_files = ['/home/zhaojin/data/TacomaBridge/segdata/train/img/00034.png' ]
    # out_files = ['/home/zhaojin/my_path/dir/segdata/predict/00025.png']
    in_files = args.input
    net = UNet(n_channels=1, n_classes=4)

    print("Loading model {}".format(args.model))

    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net.load_state_dict(torch.load(args.model))
    else:
        net.cpu()
        net.load_state_dict(torch.load(args.model, map_location='cpu'))
        print("Using CPU version of the net, this may be very slow")

    print("Model loaded !")

    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)
        if img.size[0] < img.size[1]:
            print("Error: image height larger than the width")

        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           use_dense_crf= not args.no_crf,
                           use_gpu=not args.cpu)

        if args.viz:
            print("Visualizing results for image {}, close to continue ...".format(fn))
            mask = np.transpose(mask, axes=[1,2,0])
            plot_img_and_mask(img, mask)
        # if not args.no_save:
        #     out_fn = out_files[i]
        #     print('mask', mask)
        #     result = mask_to_image(mask)
        #
        #     result.save(out_files[i])
        #
        #     print("Mask saved to {}".format(out_files[i]))

