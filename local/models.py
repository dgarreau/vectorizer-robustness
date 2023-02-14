#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Re-implementation of PVDM and PVDBOW models. 

kudos to https://github.com/inejc/paragraph-vectors

"""

# TODO (carefuly) implement 1/T

import torch
import torch.nn as nn
import time
import numpy as np
from enum import Enum

from utils import MODELS_DIR

from sys import stdout

from torch.optim import Adam
from torch.utils.data import DataLoader


from os.path import join

from .data import ContexifiedDataSet
from .loss import LogSoftmax
from .inference import compute_embedding


def _print_progress(epoch_i, batch_i, num_batches):
    progress = round((batch_i + 1) / num_batches * 100)
    print("\rEpoch {:d}".format(epoch_i + 1), end="")
    stdout.write(" - {:d}%".format(progress))
    stdout.flush()


class ParagraphVectorVariant(Enum):
    PVDBOW = 1
    PVDMmean = 2
    PVDMconcat = 3


class ParagraphVector(nn.Module):
    def __init__(self, dim=50, context_size=5, variant=ParagraphVectorVariant.PVDBOW):
        super(ParagraphVector, self).__init__()
        self.dim = dim
        self.context_size = context_size
        self.variant = variant

    def forward(self, context_ids, doc_ids):
        if self.variant == ParagraphVectorVariant.PVDMconcat:
            current_batch_size = context_ids.shape[0]

            h = torch.add(
                self._Q_matrix[doc_ids, :],
                torch.mm(
                    self.one_hot_buffer[context_ids, :].reshape(
                        current_batch_size, self._P_matrix.shape[0]
                    ),
                    self._P_matrix,
                ),
            )
        elif self.variant == ParagraphVectorVariant.PVDMmean:
            h = torch.add(
                self._Q_matrix[doc_ids, :],
                torch.sum(self._P_matrix[context_ids, :], dim=1),
            )
        elif self.variant == ParagraphVectorVariant.PVDBOW:
            h = self._Q_matrix[doc_ids, :]
        else:
            raise NotImplementedError

        # times R
        return torch.mm(h, self._R_matrix)

    def train(
        self,
        dataset,
        lr=0.002,
        n_epochs=100,
        batch_size=32,
        num_workers=8,
        verbose=False,
    ):

        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers

        num_noise_words = 0
        raw_data, vocabulary = dataset
        ctx_dataset = ContexifiedDataSet(
            raw_data,
            self.context_size,
        )
        dataloader = DataLoader(
            ctx_dataset, self.batch_size, num_workers=self.num_workers
        )
        n_batches = len(dataloader)

        self.vocabulary = vocabulary

        self.n_docs = len(raw_data)
        self.n_words = len(self.vocabulary)

        # Projection matrix P
        if self.variant == ParagraphVectorVariant.PVDMconcat:
            # size 2nuD x d in our notation
            self._P_matrix = nn.Parameter(
                torch.sqrt(
                    torch.tensor([2.0])
                    / torch.tensor([2 * self.context_size * self.n_words + self.dim])
                )
                * torch.randn(2 * self.context_size * self.n_words, self.dim),
                requires_grad=True,
            )
            self.one_hot_buffer = nn.Parameter(
                torch.eye(self.n_words), requires_grad=False
            )
        elif self.variant == ParagraphVectorVariant.PVDMmean:
            # size d x D in our notation
            self._P_matrix = nn.Parameter(
                torch.sqrt(
                    torch.tensor([2.0]) / torch.tensor([self.n_words + self.dim])
                )
                * torch.randn(self.n_words, self.dim),
                requires_grad=True,
            )
        elif self.variant == ParagraphVectorVariant.PVDBOW:
            self._P_matrix = None
        else:
            raise NotImplementedError

        # Matrix R
        # final layer
        # R in our notation, size D x d
        self._R_matrix = nn.Parameter(
            torch.sqrt(torch.tensor([2.0]) / torch.tensor([self.n_words + self.dim]))
            * torch.randn(self.dim, self.n_words),
            requires_grad=True,
        )

        # embedding of the documents, Gaussian initialization
        # Q in our notation, size d x N
        self._Q_matrix = nn.Parameter(
            torch.sqrt(torch.tensor([2.0]) / torch.tensor([self.n_docs + self.dim]))
            * torch.randn(self.n_docs, self.dim),
            requires_grad=True,
        )

        if torch.cuda.is_available():
            self.cuda()
            if verbose:
                print("using cuda")
                print()

        if verbose:
            if self.variant == ParagraphVectorVariant.PVDMconcat:
                print("PVDM-concat training starts:")
            elif self.variant == ParagraphVectorVariant.PVDMmean:
                print("PVDM-mean training starts:")
            elif self.variant == ParagraphVectorVariant.PVDBOW:
                print("PVDBOW training starts:")
            else:
                raise NotImplementedError
            print("N = {:d}".format(self.n_docs))
            print("D = {:d}".format(self.n_words))
            print("d = {:d}".format(self.dim))
            print("nu = {:d}".format(self.context_size))
            print("lr = {:.4f}".format(self.lr))
            print()

        # loss function
        cost_func = LogSoftmax()

        # optimizer
        optimizer = Adam(params=self.parameters(), lr=self.lr)

        # entering the main loop
        t_start = time.time()
        loss_values = []
        for i_epoch in range(n_epochs):
            epoch_start = time.time()
            loss = []
            i_batch = 0  # enumerate is bad for multiprocessing?
            for batch in dataloader:
                context_ids, doc_ids, target_ids = batch
                x = self.forward(context_ids, doc_ids)
                x = cost_func.forward(x, target_ids)
                loss.append(x.item())
                self.zero_grad()
                x.backward()
                optimizer.step()
                if verbose:
                    _print_progress(i_epoch, i_batch, n_batches)
                i_batch += 1

            # end of epoch
            loss = torch.mean(torch.FloatTensor(loss))
            loss_values.append(loss)

            epoch_end = time.time()

            epoch_total_time = round(epoch_end - epoch_start)
            if verbose:
                print(" ({:d}s) - loss: {:.4f}".format(epoch_total_time, loss))
        t_end = time.time()

        # saving relevant info
        self.loss_values = loss_values
        if verbose:
            print("elapsed = {}s".format(np.round(t_end - t_start, 0)))

    def get_P_matrix(self):
        """
        Get P matrix as numpy array.
        """
        return self._P_matrix.cpu().detach().numpy().T

    def get_Q_matrix(self):
        """
        Get Q matrix as numpy array.
        """
        return self._Q_matrix.cpu().detach().numpy().T

    def get_R_matrix(self):
        """
        Get R matrix as numpy array.
        """
        return self._R_matrix.cpu().detach().numpy().T

    def load(self, name, verbose=False):
        """
        Loading a model from the MODELS_DIR folder.

        BEWARE: sizes have to match.
        """

        vocab_name = "vocab_" + name
        param_name = "params_" + name

        model_file_path = join(MODELS_DIR, name)
        vocab_file_path = join(MODELS_DIR, vocab_name)
        param_file_path = join(MODELS_DIR, param_name)

        if verbose:
            print("loading model from {}".format(model_file_path))
            print()

        self.vocabulary = torch.load(vocab_file_path)

        param_dict = torch.load(param_file_path)
        self.context_size = param_dict["nu"]
        self.variant = param_dict["variant"]

        # load the model
        aux = torch.load(model_file_path)

        # two possibilities depending if saved during training or after
        if "_P_matrix" in aux.keys():
            self._P_matrix = nn.Parameter(aux["_P_matrix"])
            self._Q_matrix = nn.Parameter(aux["_Q_matrix"])
            self._R_matrix = nn.Parameter(aux["_R_matrix"])
        else:
            self._P_matrix = nn.Parameter(aux["model_state_dict"]["_P_matrix"])
            self._Q_matrix = nn.Parameter(aux["model_state_dict"]["_Q_matrix"])
            self._R_matrix = nn.Parameter(aux["model_state_dict"]["_R_matrix"])

        self.n_words = len(self.vocabulary)
        self.n_docs, self.dim = self._Q_matrix.shape

        if verbose:
            print("D = {}".format(self.n_words))
            print("d = {}".format(self.dim))
            print("context_size = {}".format(self.context_size))
            print("variant = {}".format(self.variant))
            print()

    def save(self, name, verbose=False):
        """
        Saving the model.

        BEWARE: simply saving the weights, distinct from the save_training_state util.
        """

        vocab_name = "vocab_" + name
        param_name = "params_" + name

        model_file_path = join(MODELS_DIR, name)
        vocab_file_path = join(MODELS_DIR, vocab_name)
        param_file_path = join(MODELS_DIR, param_name)

        if verbose:
            print("saving weights at {}".format(model_file_path))
            print("saving vocab at {}".format(vocab_file_path))
            print("saving parameters at {}".format(param_file_path))
            print()

        # saving some additional parameters
        param_dict = {}
        param_dict["nu"] = self.context_size
        # param_dict['lr'] = self.lr
        param_dict["variant"] = self.variant

        torch.save(self.state_dict(), model_file_path)
        torch.save(self.vocabulary, vocab_file_path)
        torch.save(param_dict, param_file_path)

    def infer(self, document, n_steps=None, gamma=None, track_objective=False):
        """
        Get embedding.
        """
        P_array = self.get_P_matrix()
        R_array = self.get_R_matrix()
        tokenized_doc = np.array([self.vocabulary.stoi[w] for w in document], dtype=int)
        q_vec, traj_store, obj_store = compute_embedding(
            tokenized_doc,
            R_array,
            model=self.variant,
            P_matrix=P_array,
            mode="true",
            track_objective=track_objective,
            winsize=self.context_size,
            n_steps=n_steps,
            gamma=gamma,
        )
        return q_vec
