## test for split
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from unet import UNet
from utils import resize_and_crop, normalize, split_img_into_squares_batch, hwc_to_chw, merge_masks, dense_crf,resize_np
from utils import plot_img_and_mask
from utils import Dataset_predict
from torchvision import transforms

import matplotlib.pyplot as plt
from torch.utils import data



def predict_img_batch(net,
                imgpath,
                lblpath,
                batchsize=10,
                scale_factor=0.5,
                out_threshold=0.5,
                use_dense_crf=True,
                use_gpu=True):
    """return fullmask with size (C, H, W)"""
    net.eval()
    print('imgpath', imgpath )
    predictdataset = Dataset_predict(imgpath, "L", 0.5)
    predictdatagen = data.DataLoader(predictdataset,batchsize)

    for iB, (img, imgids) in enumerate(predictdatagen):
        print('batch num {}'.format(iB))
        # print('img.shape', img.shape )
        """ img with size (N H W C)"""
        # print('np.max(img))', np.max(np.array(img)))
        # img = normalize(img)
        # print('np.max(img))', np.max(np.array(img)))
        left_square, right_square = split_img_into_squares_batch(img)
        img_width = int(img.shape[2]/scale_factor)
        "(N H W C) --->  (N C H W)"
        # print('left_square.shape', left_square.shape )
        left_square = np.transpose(left_square, [0, 3, 1, 2]) # (N C H W)
        right_square = np.transpose(right_square, [0, 3, 1, 2])
        # print('type(left_square)', type(left_square) )
        # X_left = torch.from_numpy(left_square)
        # X_right = torch.from_numpy(right_square)

        X_left = left_square.type(torch.FloatTensor)
        X_right = right_square.type(torch.FloatTensor)
        if use_gpu:
            X_left = X_left.cuda()
            X_right = X_right.cuda()

        with torch.no_grad():
            output_left = net(X_left)  # (N C H W)
            output_right = net(X_right)


            print('output_left.shape', output_left.shape)
            print('output_right.shape', output_right.shape)
            left_probs = output_left  # (N C H W)
            right_probs = output_right

            left_mask_np = left_probs.cpu().numpy()  # (N C H W)
            right_mask_np = right_probs.cpu().numpy()  # (N C H W)

            for iN, imgid in enumerate(imgids):
                left_mask_np_n = left_mask_np[iN]  # ( C H W)
                right_mask_np_n = right_mask_np[iN]
                left_mask_np_n = np.transpose(left_mask_np_n, axes=[1, 2, 0])  # (H W C)
                right_mask_np_n = np.transpose(right_mask_np_n, axes=[1, 2, 0])
                if not scale_factor == 1:
                    right_mask_np_n = resize_np(right_mask_np_n, 1/scale_factor)  # (H/2 W/2 C)
                    left_mask_np_n = resize_np(left_mask_np_n, 1/scale_factor)
                right_mask_np_n = np.transpose(right_mask_np_n, axes=[2,0,1])  # (C H W)
                left_mask_np_n = np.transpose(left_mask_np_n, axes=[2,0,1])
                print('left_mask_np.shape_n', left_mask_np_n.shape )
                print('img_width', img_width )
                full_mask = merge_masks(left_mask_np_n, right_mask_np_n, img_width)
                full_mask = np.transpose(full_mask, axes=[1,2,0])  # (H W C)
                print('full_mask.shape', full_mask.shape )
                np.savez_compressed(lblpath + '/' + imgid.split('.png')[0] + '_mask.npz', label=full_mask)  ## default name is arr_0

    #
    #
    #     left_mask_np = np.transpose(left_mask_np, axes=[2, 0, 1])
    #     right_mask_np = np.transpose(right_mask_np, axes=[2, 0, 1])
    #     full_mask = merge_masks(left_mask_np, right_mask_np, img_width)
    #
    # # if use_dense_crf:
    # if 0:
    #     full_mask = dense_crf(np.array(full_img).astype(np.uint8), full_mask)
    #
    # return full_mask
    # # return left_mask_np

def get_args(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')")
    parser.add_argument('--input', '-i', metavar='INPUT',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT',
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

    return parser.parse_args(raw_args)

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


def predict_5outpout(raw_args=None):
    """example:  python predict_batch.py --model '/home/zhaojin/data/TacomaBridge/segdata/train/checkpoint/weight_logloss_softmax/CP30.pth' --input '/home/zhaojin/data/TacomaBridge/capture/high-reso-clip2_rename' --output '/home/zhaojin/data/TacomaBridge/segdata/predict/high-reso-clip2_rename'"""
    args = get_args(raw_args)
    # args.model = '/home/zhaojin/data/TacomaBridge/segdata/train/checkpoint/logloss_softmax/CP12.pth'
    # in_files = ['/home/zhaojin/data/TacomaBridge/segdata/train/img/00034.png' ]
    # out_files = ['/home/zhaojin/my_path/dir/segdata/predict/00025.png']
    imgpath = args.input
    lblpath = args.output
    net = UNet(n_channels=1, n_classes=5)

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


    predict_img_batch(net=net,
                            imgpath=imgpath,
                            lblpath=lblpath,
                            scale_factor=args.scale,
                            out_threshold=args.mask_threshold,
                            use_dense_crf= not args.no_crf,
                            use_gpu=not args.cpu)

    # if args.viz:
    #     print("Visualizing results for image {}, close to continue ...".format(fn))
    #     mask = np.transpose(mask, axes=[1,2,0])
    #     plot_img_and_mask(img, mask)

if __name__ == "__main__":
    main()