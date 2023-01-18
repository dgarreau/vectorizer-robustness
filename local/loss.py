#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Loss function, without fancy stuff.

"""

import torch
import torch.nn as nn

class LogSoftmax(nn.Module):
    
    def __init__(self):
        super(LogSoftmax, self).__init__()
        self._log_sigmoid = nn.LogSoftmax(dim=1)

    def forward(self,scores,target_ids):

        batch_size,_ = scores.shape
        
        aux = -self._log_sigmoid(scores)

        probas = aux[torch.arange(batch_size),torch.transpose(target_ids,0,1)]

        return torch.sum(probas)


