import torch
import torch.nn as nn
import numpy as np
from BCELoss_weight import BCELoss_weight

tarnp = np.zeros((10,5,12,12), dtype=np.float32)

target = torch.from_numpy(tarnp)
target = torch.FloatTensor(target)

output = torch.randn(10, 5,12,12, requires_grad=True)
m = nn.Sigmoid()
output = m(output)
print('output.shape', output.shape )
print('output.shape', output.shape )
criterion = BCELoss_weight(weight=[1 ,2,3,4])
loss = criterion(output, target)
loss.backward()