from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch

import numpy as np
import torch

from my_dice import dice_cofe


def eval_net(net, dataset):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    for i, b in enumerate(dataset):
        img = b[0]
        true_mask = b[1]

        true_mask = np.transpose(true_mask, axes=[2,0,1])
        img = torch.from_numpy(img).unsqueeze(0)
        # true_mask = torch.from_numpy(true_mask)


        mask_pred = net(img.cuda())[0]
        mask_pred = mask_pred.data.cpu().numpy()
        masksum = np.sum(mask_pred,axis=0)

        print('np.max(masksum)', np.max(masksum) )
        print('np.min(masksum)', np.min(masksum) )


    return mask_pred

if __name__ == "__main__":

    dir_img = '/home/zhaojin/data/TacomaBridge/segdata/train/img'
    dir_mask = '/home/zhaojin/data/TacomaBridge/segdata/train/mask'
    dir_checkpoint = '/home/zhaojin/data/TacomaBridge/segdata/train/checkpoint/logloss_softmax/'


    net = UNet(n_channels=1, n_classes=4)
    net = net.cuda()
    ids = get_ids(dir_img)
    ids = split_ids(ids)

    iddataset = split_train_val(ids,0.1)


    val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, 0.5)
    if 1:
        val_dice = eval_net(net, val)
        print('Validation Dice Coeff: {}'.format(val_dice))