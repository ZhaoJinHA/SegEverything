
import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
from BCELoss_weight import BCELoss_weight

def train_net(net,
              epochs=20,
              batch_size=1,
              lr=0.1,
              lrd=0.99,
              val_percent=0.05,
              save_cp=True,
              gpu=True,
              img_scale=0.5,
              imagepath='',
              maskpath='',
              cpsavepath=''):

    dir_img = imagepath
    dir_mask = maskpath
    dir_checkpoint = cpsavepath
    classweight = [1, 2, 3, 2, 3]

    ids = get_ids(dir_img)
    ids = split_ids(ids)

    iddataset = split_train_val(ids, val_percent)

    logname = cpsavepath + '/' + 'losslog.txt'

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu)))

    N_train = len(iddataset['train'])

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    # classweight = [1,4,8,4]
    criterion = BCELoss_weight(classweight)

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        with open(logname, "a") as f:
            f.write('Starting epoch {}/{}.'.format(epoch + 1, epochs) + "\n")
        net.train()

        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)

        epoch_loss = 0
        
        lr = lr*lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('lr', lr)
        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])
            
            true_masks = np.transpose(true_masks, axes=[0,3,1,2])
            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs)
            # print('masks_pred.shape',masks_pred.shape)
            # print('true_masks.shape', true_masks.shape)
            masks_probs_flat = masks_pred

            true_masks_flat = true_masks
            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()

            printinfo = '{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item())
            print(printinfo)

            with open(logname, "a") as f:
                f.write(printinfo + "\n")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))
        with open(logname, "a") as f:
            f.write('Epoch finished ! Loss: {}'.format(epoch_loss / i) + "\n")
        if 1:
            val_dice = eval_net(net, val)
            print('Validation Dice Coeff: {}'.format(val_dice))
            with open(logname, "a") as f:
                f.write('Validation Dice Coeff: {}'.format(val_dice) + "\n")

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))
            with open(logname, "a") as f:
                f.write('Checkpoint {} saved !'.format(epoch + 1) + "\n")



def get_args(raw_args=None):
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=60, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-d', '--learning-rate-damping', dest='lrd', default=0.99,
                      type='float', help='learning rate damping')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')
    parser.add_option("-i", "--imagepath", default= '/home/zhaojin/data/TacomaBridge/segdata/train/img',
                      action="store", type="string", dest="imagepath")
    parser.add_option("-m", "--maskpath", default= '/home/zhaojin/data/TacomaBridge/segdata/train/mask',
                      action="store", type="string", dest="maskpath")
    parser.add_option("-v", "--checkpointsavepath", default= '/home/zhaojin/data/TacomaBridge/segdata/train/checkpoint/weight_logloss_softmax_test/',
                      action="store", type="string", dest="savepath")
    (options, args) = parser.parse_args(raw_args)
    return options

def train_5output(raw_args=None):
    args = get_args(raw_args)
    # args.epochs = 30
    # args.lr = 0.1
    save_cp = True
    args.load = ''
    # args.batchsize=20
    # args.scale=0.5
    imagepath=args.imagepath
    maskpath=args.maskpath
    cpsavepath = args.savepath
    args.gpu = True
    net = UNet(n_channels=1, n_classes=5)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  lrd=args.lrd,
                  gpu=args.gpu,
                  save_cp = save_cp,
                  img_scale=args.scale,
                  imagepath=imagepath,
                  maskpath=maskpath,
                  cpsavepath=cpsavepath)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


   
    
    
def train_resume():
    args = get_args()
    args.epochs = 100
    args.lr = 0.0075
    save_cp = True
    args.load = '/home/zhaojin/data/TacomaBridge/segdata/train/checkpointCP30.pth'
    
    args.gpu = True
    net = UNet(n_channels=1, n_classes=4)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  save_cp = save_cp,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

if __name__ == '__main__':
    # train_1st()


    inputcode = ['-i', '/home/zhaojin/data/TacomaBridge/segdata/girder2cls_data_2ep/img', '-m', '/home/zhaojin/data/TacomaBridge/segdata/girder2cls_data_2ep/mask', '-v',
                 '/home/zhaojin/data/TacomaBridge/segdata/girder2cls_data_2ep/']
    train_5output(inputcode)
