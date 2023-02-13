#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data interface, kudos to https://github.com/inejc/paragraph-vectors

"""

import torch
import numpy as np

from numpy.random import choice

from math import floor

from torch.utils.data import DataLoader



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
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True)
        self.dataloader = iter(self.dataloader)
        self._vocabulary = dataset["vocab"]
        self._counter = dataset["counter"]
        self._tokenizer = dataset["tokenizer"]
        self._sample_noise = None
        self._prev_doc_id = 0
        self._prev_in_doc_pos = self.context_size
        self._init_noise_distribution()

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
        return floor(num_examples / self.batch_size)

    def vocabulary_size(self):
        return len(self._vocabulary) - 1

    def __iter__(self):
        return self

    def __next__(self):
        """Updates state for the next process in a process-safe manner
        and generates the current batch."""

        num_examples = self._num_examples_in_doc(
            self.dataset[self._prev_doc_id], self._prev_in_doc_pos
        )

        if num_examples > self.batch_size:
            # more examples in the current document
            self._prev_in_doc_pos += self.batch_size

        elif num_examples == self.batch_size:
            # just enough examples in the current document
            if self._prev_doc_id < len(self.dataset) - 1:
                self._prev_doc_id += 1
            else:
                self._prev_doc_id = 0
            self._prev_in_doc_pos = self.context_size

        else:
            while num_examples < self.batch_size:
                if self._prev_doc_id == len(self.dataset) - 1:
                    # last document: reset indices
                    self._prev_doc_id = 0
                    self._prev_in_doc_pos = self.context_size
                    break

                self._prev_doc_id += 1
                num_examples += self._num_examples_in_doc(self.dataset[self._prev_doc_id])

            last_doc = self._tokenizer(self.dataset[self._prev_doc_id][1])
            self._prev_in_doc_pos = (
                len(last_doc)
                - self.context_size
                - (num_examples - self.batch_size)
            )


        # generate the actual batch
        #batch = next(self.dataloader)
        context_ids = [] if self.context_size > 0 else None
        doc_ids = []
        target_noise_ids = []

        while len(doc_ids) < self.batch_size:
            if self._prev_doc_id == len(self.dataset):
                # last document exhausted
                return (
                    torch.LongTensor(context_ids),
                    torch.LongTensor(doc_ids),
                    torch.LongTensor(target_noise_ids),
                )
            len_prev_doc = len(self._tokenizer(self.dataset[self._prev_doc_id][1]))
            if self._prev_in_doc_pos <= (
                len_prev_doc - 1 - self.context_size
            ):
                # more examples in the current document
                self._add_example_to_batch(self._prev_doc_id, self._prev_in_doc_pos, context_ids, doc_ids, target_noise_ids)
                self._prev_in_doc_pos += 1
            else:
                # go to the next document
                self._prev_doc_id += 1
                self._prev_in_doc_pos = self.context_size

        return (
            torch.LongTensor(context_ids),
            torch.LongTensor(doc_ids),
            torch.LongTensor(target_noise_ids),
        )


    def _num_examples_in_doc(self, doc, in_doc_pos=None):
        doc_t = self._tokenizer(doc[1])
        if in_doc_pos is not None:
            # number of remaining
            if len(doc_t) - in_doc_pos >= self.context_size + 1:
                return len(doc_t) - in_doc_pos - self.context_size
            return 0

        if len(doc[1]) >= 2 * self.context_size + 1:
            # total number
            return len(doc_t) - 2 * self.context_size
        return 0

    def _add_example_to_batch(self, doc_id, in_doc_pos, context_ids, doc_ids, target_noise_ids):
        doc = self.dataset[doc_id][1]
        doc = self._tokenizer(doc)
        doc_ids.append(doc_id)

        # sample from the noise distribution
        current_noise = self._sample_noise()
        current_noise.insert(0, self._word_to_index(doc[in_doc_pos]))
        target_noise_ids.append(current_noise)

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
        context_ids.append(current_context)

    def _word_to_index(self, word):
        return self._vocabulary[word] - 1


# class _NCEGeneratorState:
#     """Batch generator state that is represented with a document id and
#     in-document position. It abstracts a process-safe indexing mechanism."""

#     def __init__(self, context_size):
#         # use raw values because both indices have
#         # to manually be locked together
#         self._doc_id = 0
#         self._in_doc_pos = context_size

#     def update_state(self, dataset, batch_size, context_size, num_examples_in_doc):
#         """Returns current indices and computes new indices for the
#         next process."""
#         doc_id = self._doc_id
#         in_doc_pos = self._in_doc_pos
#         self._advance_indices(
#             dataset, batch_size, context_size, num_examples_in_doc
#         )
#         return doc_id, in_doc_pos

#     def _advance_indices(self, dataset, batch_size, context_size, num_examples_in_doc):
#         num_examples = num_examples_in_doc(
#             dataset[self._doc_id], self._in_doc_pos
#         )

#         if num_examples > batch_size:
#             # more examples in the current document
#             self._in_doc_pos += batch_size
#             return

#         if num_examples == batch_size:
#             # just enough examples in the current document
#             if self._doc_id < len(dataset) - 1:
#                 self._doc_id += 1
#             else:
#                 self._doc_id = 0
#             self._in_doc_pos = context_size
#             return

#         while num_examples < batch_size:
#             if self._doc_id == len(dataset) - 1:
#                 # last document: reset indices
#                 self._doc_id = 0
#                 self._in_doc_pos = context_size
#                 return

#             self._doc_id += 1
#             num_examples += num_examples_in_doc(dataset[self._doc_id])

#         self._in_doc_pos = (
#             len(dataset[self._doc_id][1])
#             - context_size
#             - (num_examples - batch_size)
#         )


# class _NCEBatch:
#     def __init__(self, context_size):
#         self.context_ids = [] if context_size > 0 else None
#         self.doc_ids = []
#         self.target_noise_ids = []

#     def __len__(self):
#         return len(self.doc_ids)

#     def torch_(self):
#         if self.context_ids is not None:
#             self.context_ids = torch.LongTensor(self.context_ids)
#         self.doc_ids = torch.LongTensor(self.doc_ids)
#         self.target_noise_ids = torch.LongTensor(self.target_noise_ids)

#     def cuda_(self):
#         if self.context_ids is not None:
#             self.context_ids = self.context_ids.cuda()
#         self.doc_ids = self.doc_ids.cuda()
#         self.target_noise_ids = self.target_noise_ids.cuda()
