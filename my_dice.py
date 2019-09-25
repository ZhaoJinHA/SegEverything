import numpy as np

def dice_cofe(input, target):
    """ input and target need with same size, and they are numpy array"""
    input = np.array(input).astype(int)
    target = np.array(target).astype(int)
    input = input.reshape(-1)
    target = target.reshape(-1)
    
    return np.sum(input[target==1])*2.0 / (np.sum(input) + np.sum(target))
