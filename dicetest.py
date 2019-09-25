
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

import torch
from torch.autograd import Function, Variable

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps
        # print('input.shape', input.shape)
        # print('target.shape', target.shape)
        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


npzfilename = '/home/zhaojin/data/TacomaBridge/segdata/train/mask/00034_mask.npz'
nparr0 = np.load(npzfilename)['label']
nparr0 = np.array(nparr0).astype(np.float32)
nparr0 = np.transpose(nparr0, axes=[2,0,1])
npzfilename = '/home/zhaojin/data/TacomaBridge/segdata/train/testoutput.npz'
nparr = np.load(npzfilename)['label']
nparr = np.array(nparr).astype(np.float32)
nparr = torch.from_numpy(nparr)
nparr0 = torch.from_numpy(nparr0)
print('nparr.shape', nparr.shape)
print('nparr0.shape', nparr0.shape)
dicec = dice_coeff(nparr0, nparr0).item()
print('dicec', dicec)