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



class ContexifiedDataSet(Dataset):
    """ Take an arbitrary dataset of documents and transform it chunk of contexts
    """

    def __init__(self, dataset, ctx_size, tokenizer, vocabulary, counter, num_noise_words = 0, txt_idx=1):
        self.dataset = dataset
        self.num_noise_words = num_noise_words
        self._tokenizer = tokenizer
        self._vocabulary = vocabulary
        self._counter = counter
        # HACK: transform into list the original dataset. BAD if large.
        self.data = list(map(lambda e: torch.tensor([self._vocabulary[token] for token in self._tokenizer(e[txt_idx])]), list(self.dataset)))
        self.ctx_size = ctx_size
        self._sample_noise = self._init_noise_distribution()

        self._number_doc = len(self.data)

        # Compute the number of examples in the dataset
        self._number_examples = torch.tensor([
            self._number_examples_in_doc(doc_id)
            for doc_id in range(self._number_doc)
        ])
        self._cumulative_number_examples = torch.cumsum(self._number_examples,dim=0)
        self._total_number_examples = torch.sum(self._number_examples)

    def __len__(self):
        return self._total_number_examples

    def __getitem__(self, idx):
        doc_id = torch.searchsorted(self._cumulative_number_examples, idx+1)
        doc_t = self.data[doc_id]
        ctx_idx = idx - self._cumulative_number_examples[doc_id-1] if doc_id > 0 else idx
        return (self._ctx_at(doc_t, ctx_idx), doc_id, self._sample_noise())


    def _ctx_at(self, doc, ctx_id):
        return torch.cat((doc[ctx_id:(ctx_id+self.ctx_size)], doc[(ctx_id+self.ctx_size+1):(ctx_id+2*self.ctx_size+1)]))

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

    def _init_noise_distribution(self):
        # we use a unigram distribution raised to the 3/4rd power,
        # as proposed by T. Mikolov et al. in Distributed Representations
        # of Words and Phrases and their Compositionality
        probs = np.zeros(len(self._vocabulary) - 1)

        for word in self._counter:
            probs[self._vocabulary[word]-1] = self._counter[word] # frequency

        probs = np.power(probs, 0.75)
        probs /= np.sum(probs)

        return (lambda: torch.tensor(choice(probs.shape[0], self.num_noise_words, p=probs)))
