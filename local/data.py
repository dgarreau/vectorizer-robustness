#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data interface, kudos to https://github.com/inejc/paragraph-vectors

"""

import torch
import numpy as np

from numpy.random import choice

from math import ceil

from torch.utils.data import DataLoader, Dataset

class MultiEpochsDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


# NOTE i tried to cache contexts but the speed improvement is low
class ContexifiedDataSet(Dataset):
    """Take an arbitrary dataset of documents and transform it chunk of contexts"""

    def __init__(self, data, ctx_size):
        self.data = data
        self.ctx_size = ctx_size

        self._number_doc = len(self.data)

        # Compute the number of examples in the dataset
        self._number_examples = torch.tensor(
            [self._number_examples_in_doc(doc_id) for doc_id in range(self._number_doc)]
        )
        self._cumulative_number_examples = torch.cumsum(self._number_examples, dim=0)
        self._total_number_examples = torch.sum(self._number_examples)

    def __len__(self):
        return self._total_number_examples

    def __getitem__(self, idx):
        doc_id = torch.searchsorted(self._cumulative_number_examples, idx + 1)
        doc_t = self.data[doc_id]
        ctx_idx = (
            idx - self._cumulative_number_examples[doc_id - 1] if doc_id > 0 else idx
        )
        elem = (self._ctx_at(doc_t, ctx_idx), doc_id, doc_t[ctx_idx + self.ctx_size])
        return elem

    def _ctx_at(self, doc, ctx_id):
        return torch.cat(
            (
                doc[ctx_id : (ctx_id + self.ctx_size)],
                doc[(ctx_id + self.ctx_size + 1) : (ctx_id + 2 * self.ctx_size + 1)],
            )
        )

    def _number_examples_in_doc(self, doc_id, in_doc_pos=None):
        doc_t = self.data[doc_id]
        num = 0
        if in_doc_pos:
            # Remaining contexts starting at in_doc_pos
            num = max(0, len(doc_t) - in_doc_pos - self.ctx_size)
        else:
            # Total number of contexts in doc_t
            num = max(0, len(doc_t) - 2 * self.ctx_size)
        return num
