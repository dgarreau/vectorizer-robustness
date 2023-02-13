#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data interface, kudos to https://github.com/inejc/paragraph-vectors

"""

import torch
import numpy as np

from numpy.random import choice

from math import ceil


class NCEGenerator:
    """An infinite, process-safe batch generator for noise-contrastive
    estimation of word vector models.
    Parameters
    ----------
    state: paragraphvec.data._NCEGeneratorState
        Initial (indexing) state of the generator.
    For other parameters see the NCEData class.
    """

    def __init__(self, dataset, batch_size, context_size, num_noise_words, state=None):
        self.dataset = dataset["dataset"]
        self.batch_size = batch_size
        self.context_size = context_size
        self.num_noise_words = num_noise_words

        self._vocabulary = dataset["vocab"]
        self._counter = dataset["counter"]
        self._sample_noise = None
        self._init_noise_distribution()
        if not state:
            self._state = _NCEGeneratorState(context_size)

    def _init_noise_distribution(self):
        # we use a unigram distribution raised to the 3/4rd power,
        # as proposed by T. Mikolov et al. in Distributed Representations
        # of Words and Phrases and their Compositionality
        probs = np.zeros(len(self._vocabulary) - 1)

        for word in self._counter:
            probs[self._word_to_index(word)] = self._counter[word] # frequency

        probs = np.power(probs, 0.75)
        probs /= np.sum(probs)

        self._sample_noise = lambda: choice(
            probs.shape[0], self.num_noise_words, p=probs
        ).tolist()

    def __len__(self):
        num_examples = sum(self._num_examples_in_doc(d) for d in self.dataset)
        return ceil(num_examples / self.batch_size)

    def vocabulary_size(self):
        return len(self._vocabulary) - 1

    def __iter__(self):
        return self

    def __next__(self):
        """Updates state for the next process in a process-safe manner
        and generates the current batch."""
        prev_doc_id, prev_in_doc_pos = self._state.update_state(
            self.dataset, self.batch_size, self.context_size, self._num_examples_in_doc
        )

        # generate the actual batch
        batch = _NCEBatch(self.context_size)

        while len(batch) < self.batch_size:
            if prev_doc_id == len(self.dataset):
                # last document exhausted
                batch.torch_()
                return batch
            if prev_in_doc_pos <= (
                len(self.dataset[prev_doc_id].text) - 1 - self.context_size
            ):
                # more examples in the current document
                self._add_example_to_batch(prev_doc_id, prev_in_doc_pos, batch)
                prev_in_doc_pos += 1
            else:
                # go to the next document
                prev_doc_id += 1
                prev_in_doc_pos = self.context_size

        batch.torch_()
        return batch

    def _num_examples_in_doc(self, doc, in_doc_pos=None):
        if in_doc_pos is not None:
            # number of remaining
            if len(doc.text) - in_doc_pos >= self.context_size + 1:
                return len(doc.text) - in_doc_pos - self.context_size
            return 0

        if len(doc.text) >= 2 * self.context_size + 1:
            # total number
            return len(doc.text) - 2 * self.context_size
        return 0

    def _add_example_to_batch(self, doc_id, in_doc_pos, batch):
        doc = self.dataset[doc_id].text
        batch.doc_ids.append(doc_id)

        # sample from the noise distribution
        current_noise = self._sample_noise()
        current_noise.insert(0, self._word_to_index(doc[in_doc_pos]))
        batch.target_noise_ids.append(current_noise)

        if self.context_size == 0:
            return

        current_context = []
        context_indices = (
            in_doc_pos + diff
            for diff in range(-self.context_size, self.context_size + 1)
            if diff != 0
        )

        for i in context_indices:
            context_id = self._word_to_index(doc[i])
            current_context.append(context_id)
        batch.context_ids.append(current_context)

    def _word_to_index(self, word):
        return self._vocabulary[word] - 1


class _NCEGeneratorState:
    """Batch generator state that is represented with a document id and
    in-document position. It abstracts a process-safe indexing mechanism."""

    def __init__(self, context_size):
        # use raw values because both indices have
        # to manually be locked together
        self._doc_id = 0
        self._in_doc_pos = context_size

    def update_state(self, dataset, batch_size, context_size, num_examples_in_doc):
        """Returns current indices and computes new indices for the
        next process."""
        doc_id = self._doc_id
        in_doc_pos = self._in_doc_pos
        self._advance_indices(
            dataset, batch_size, context_size, num_examples_in_doc
        )
        return doc_id, in_doc_pos

    def _advance_indices(self, dataset, batch_size, context_size, num_examples_in_doc):
        num_examples = num_examples_in_doc(
            dataset[self._doc_id], self._in_doc_pos
        )

        if num_examples > batch_size:
            # more examples in the current document
            self._in_doc_pos += batch_size
            return

        if num_examples == batch_size:
            # just enough examples in the current document
            if self._doc_id < len(dataset) - 1:
                self._doc_id += 1
            else:
                self._doc_id = 0
            self._in_doc_pos = context_size
            return

        while num_examples < batch_size:
            if self._doc_id == len(dataset) - 1:
                # last document: reset indices
                self._doc_id = 0
                self._in_doc_pos = context_size
                return

            self._doc_id += 1
            num_examples += num_examples_in_doc(dataset[self._doc_id])

        self._in_doc_pos = (
            len(dataset[self._doc_id].text)
            - context_size
            - (num_examples - batch_size)
        )


class _NCEBatch:
    def __init__(self, context_size):
        self.context_ids = [] if context_size > 0 else None
        self.doc_ids = []
        self.target_noise_ids = []

    def __len__(self):
        return len(self.doc_ids)

    def torch_(self):
        if self.context_ids is not None:
            self.context_ids = torch.LongTensor(self.context_ids)
        self.doc_ids = torch.LongTensor(self.doc_ids)
        self.target_noise_ids = torch.LongTensor(self.target_noise_ids)

    def cuda_(self):
        if self.context_ids is not None:
            self.context_ids = self.context_ids.cuda()
        self.doc_ids = self.doc_ids.cuda()
        self.target_noise_ids = self.target_noise_ids.cuda()
