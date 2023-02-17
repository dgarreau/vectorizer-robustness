#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Loss function. 

"""

import torch
import torch.nn as nn


class LogSoftmax(nn.Module):
    def __init__(self):
        super(LogSoftmax, self).__init__()
        self._log_sigmoid = nn.LogSoftmax(dim=1)

    def forward(self, scores, target_ids, lengths):

        batch_size, _ = scores.shape

        aux = -self._log_sigmoid(scores)

        # import pdb; pdb.set_trace()
        #probas = aux[torch.arange(batch_size), torch.transpose(target_ids, 0, 1)]
        #HACK: remove transpose
        logit = aux[torch.arange(batch_size), target_ids]

        return torch.sum(logit/lengths)
