# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import random
import numpy as np

def initial(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

class RevisedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, delta=0.5, offset=0.3):

        super(RevisedFocalLoss, self).__init__()
        self.alpha = alpha
        self.delta = delta
        self.offset = offset

    def forward(self, outputs, targets):
        # important to add reduction='none' to keep per-batch-item loss
        ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none') 
        
        pt = torch.exp(-ce_loss)
        
        weight = torch.exp(- (pt - self.offset) ** 2 / self.delta)

        mfocal_loss =  (self.alpha * weight * ce_loss).mean()
            
        return mfocal_loss


if __name__ == '__main__':
    initial(0)
    outputs = torch.tensor([[1., 1., 1.],
                            [2.5, 7.5, 5.0],
                            [4., 2., 6.],
                            [2., 1., 4.],
                            ])
    
    targets = torch.tensor([0, 2, 0, 1])
#
    fl= RevisedFocalLoss()
##    
    loss = fl(outputs, targets)
    print(loss)
