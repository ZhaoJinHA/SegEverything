import torch
import torch.nn as nn
import torch.nn.functional as F

class BCELoss_weight(nn.Module):
    def __init__(self, weight):
        super().__init__()

        self.weight = weight
        self.nclasses = len(weight)


    def forward(self, prey, target):
        """prey and target with size (N, C, H, W)"""
        loss = 0
        for iclass in range(self.nclasses):
            loss0 = F.binary_cross_entropy(prey[:, iclass, ...], target[:, iclass, ...])
            # print('loss0', loss0 )
            loss +=  loss0* self.weight[iclass]
        return loss

