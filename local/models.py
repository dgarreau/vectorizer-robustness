#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Re-implementation of PVDM and PVDBOW models. 

kudos to https://github.com/inejc/paragraph-vectors

"""

# TODO Abstract PVDBOW / PDVM in one class and derive it
# TODO (carefuly) implement 1/T

import torch
import torch.nn as nn
import time
import numpy as np

from utils import MODELS_DIR

from sys import stdout

from torch.optim import Adam
from torch.utils.data import DataLoader


# from torchtext.legacy.data import Example,Field,Dataset

from os.path import join

from .data import NCEGenerator
from .loss import LogSoftmax
from .inference import compute_embedding

_DM_WEIGHTS_NAME = "DM_weights_N.{:d}_D.{:d}_d.{:d}_nu.{:d}_concat.{:d}.pth.tar"
_DM_VOCAB_NAME = "DM_vocab_N.{:d}_D.{:d}_d.{:d}_nu.{:d}_concat.{:d}.pth.tar"
_DM_PARAM_NAME = "DM_param_N.{:d}_D.{:d}_d.{:d}_nu.{:d}_concat.{:d}.pth.tar"

_DBOW_WEIGHTS_NAME = "DBOW_weights_N.{:d}_D.{:d}_d.{:d}_nu.{:d}.pth.tar"
_DBOW_VOCAB_NAME = "DBOW_vocab_N.{:d}_D.{:d}_d.{:d}_nu.{:d}.pth.tar"
_DBOW_PARAM_NAME = "DBOW_param_N.{:d}_D.{:d}_d.{:d}_nu.{:d}.pth.tar"


def _print_progress(epoch_i, batch_i, num_batches):
    progress = round((batch_i + 1) / num_batches * 100)
    print("\rEpoch {:d}".format(epoch_i + 1), end="")
    stdout.write(" - {:d}%".format(progress))
    stdout.flush()


class PVDBOW(nn.Module):
    """
    Paragraph vectors, distributed bag of words.
    """

    def __init__(self, dim=None, context_size=5):
        super(PVDBOW, self).__init__()

        if dim is not None:
            self.dim = dim
        if context_size is not None:
            self.context_size = context_size

    def forward(self, context_ids, doc_ids):
        return torch.mm(self._Q_matrix[doc_ids, :], self._R_matrix)

    def train(
        self,
        dataset,
        lr=0.002,
        n_epochs=100,
        batch_size=32,
        max_generated_batches=1,
        num_workers=8,
        verbose=False,
    ):

        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_generated_batches = max_generated_batches
        self.num_workers = num_workers

        num_noise_words = 0
        nce_generator = NCEGenerator(
            dataset,
            self.batch_size,
            self.context_size,
            num_noise_words,
        )
        n_batches = len(nce_generator)

        self.vocabulary = dataset["vocab"]

        self.n_docs = len(dataset["dataset"])
        self.n_words = len(self.vocabulary)

        # parameters of the model
        self._R_matrix = nn.Parameter(
            torch.sqrt(torch.tensor([2.0]) / torch.tensor([self.n_words + self.dim]))
            * torch.randn(self.dim, self.n_words),
            requires_grad=True,
        )
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

            print("PVDBOW training starts:")
            print("N = {:d}".format(self.n_docs))
            print("D = {:d}".format(self.n_words))
            print("d = {:d}".format(self.dim))
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

            for i_batch, batch in enumerate(nce_generator):
                context_ids, doc_ids, target_noise_ids = batch
                print(doc_ids)
                if len(doc_ids.size()) == 0:
                    x = self.forward(context_ids, doc_ids)
                    x = cost_func.forward(x, target_noise_ids)
                    loss.append(x.item())
                    self.zero_grad()
                    x.backward()
                    optimizer.step()
                    if verbose:
                        _print_progress(i_epoch, i_batch, n_batches)

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
        return None

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

    def get_n_docs(self):
        return self.n_docs

    def get_n_words(self):
        return self.n_words

    def get_context_size(self):
        return self.context_size

    def get_dim(self):
        return self.dim

    def save(self, name, verbose=False):
        """
        Saving the model.

        WARNING: simply saving the weights, distinct from the save_training_state util.
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
        #        param_dict['lr'] = self.lr

        torch.save(self.state_dict(), model_file_path)
        torch.save(self.vocabulary, vocab_file_path)
        torch.save(param_dict, param_file_path)

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

        # load the model
        aux = torch.load(model_file_path)

        # two possibilities depending if saved during training or after
        if "_Q_matrix" in aux.keys():
            self._Q_matrix = nn.Parameter(aux["_Q_matrix"])
            self._R_matrix = nn.Parameter(aux["_R_matrix"])
        else:
            self._Q_matrix = nn.Parameter(aux["model_state_dict"]["_Q_matrix"])
            self._R_matrix = nn.Parameter(aux["model_state_dict"]["_R_matrix"])

        self.n_words = len(self.vocabulary)
        self.n_docs, self.dim = self._Q_matrix.shape

        if verbose:
            print("D = {}".format(self.n_words))
            print("d = {}".format(self.dim))
            print("context_size = {}".format(self.context_size))
            print()

    def infer(self, document, n_steps=None, gamma=None, track_objective=False):
        """
        Get embedding.
        """
        tokenized_doc = np.array([self.vocabulary.stoi[w] for w in document], dtype=int)
        R_array = self.get_R_matrix()
        q_vec, traj_store, obj_store = compute_embedding(
            tokenized_doc,
            R_array,
            model="PVDBOW",
            P_matrix=None,
            mode="true",
            track_objective=track_objective,
            winsize=self.context_size,
            n_steps=n_steps,
            gamma=gamma,
        )
        return q_vec


class PVDM(nn.Module):
    """
    Paragraph Vectors Distributed Memory.

    PARAMETERS:
        dim: dimension of the embedding
        n_docs: number of documents in the dataset
        n_words: number of words in the dictionary
        concat (bool) concat vectors if True, average is False (default)

    """

    def __init__(self, dim=None, context_size=None, concat=False):

        # inherits the attributes of nn.Module
        super(PVDM, self).__init__()

        if dim is not None:
            self.dim = dim
        if context_size is not None:
            self.context_size = context_size

        self.concat = concat

    def forward(self, context_ids, doc_ids):

        if self.concat:

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
        else:
            h = torch.add(
                self._Q_matrix[doc_ids, :],
                torch.sum(self._P_matrix[context_ids, :], dim=1),
            )

        # times R
        return torch.mm(h, self._R_matrix)

    def train(
        self,
        dataset,
        lr=0.001,
        n_epochs=100,
        batch_size=32,
        max_generated_batches=1,
        num_workers=8,
        verbose=False,
    ):
        """
        Train the model on some data (not stored in the model, has to be created before).
        """

        # store all training parameters internally, except the data
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_generated_batches = max_generated_batches
        self.num_workers = num_workers

        # data stream for the training, no noise for us
        num_noise_words = 0
        nce_generator = NCEGenerator(
            dataset,
            self.batch_size,
            self.context_size,
            num_noise_words,
            self.max_generated_batches,
        )
        n_batches = len(nce_generator)

        self.vocabulary = dataset['vocab']

        self.n_docs = len(dataset)
        self.n_words = len(self.vocabulary)

        # word embeddings

        if self.concat:
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
        else:
            # size d x D in our notation
            self._P_matrix = nn.Parameter(
                torch.sqrt(
                    torch.tensor([2.0]) / torch.tensor([self.n_words + self.dim])
                )
                * torch.randn(self.n_words, self.dim),
                requires_grad=True,
            )

        # embedding of the documents, Gaussian initialization
        # Q in our notation, size d x N
        self._Q_matrix = nn.Parameter(
            torch.sqrt(torch.tensor([2.0]) / torch.tensor([self.n_docs + self.dim]))
            * torch.randn(self.n_docs, self.dim),
            requires_grad=True,
        )

        # final layer
        # R in our notation, size D x d
        self._R_matrix = nn.Parameter(
            torch.sqrt(torch.tensor([2.0]) / torch.tensor([self.n_words + self.dim]))
            * torch.randn(self.dim, self.n_words),
            requires_grad=True,
        )
        #        self._R_matrix = nn.Parameter(torch.FloatTensor(dim, n_words).zero_(), requires_grad=True)

        if torch.cuda.is_available():
            self.cuda()
            if verbose:
                print("using cuda")
                print()

        if verbose:
            if self.concat:
                print("PVDM-concat training starts:")
            else:
                print("PVDM-mean training starts:")
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

            for i_batch in range(n_batches):

                batch = next(nce_generator)

                if torch.cuda.is_available():
                    batch.cuda_()

                x = self.forward(batch.context_ids, batch.doc_ids)

                x = cost_func.forward(x, batch.target_noise_ids)

                loss.append(x.item())

                self.zero_grad()

                x.backward()

                optimizer.step()

                if verbose:
                    _print_progress(i_epoch, i_batch, n_batches)

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

    def get_n_docs(self):
        return self.n_docs

    def get_n_words(self):
        return self.n_words

    def get_context_size(self):
        return self.context_size

    def get_dim(self):
        return self.dim

    def is_concat(self):
        return self.concat

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
        self.concat = param_dict["concat"]

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
            print("concat = {}".format(self.concat))
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
        param_dict["concat"] = self.concat

        torch.save(self.state_dict(), model_file_path)
        torch.save(self.vocabulary, vocab_file_path)
        torch.save(param_dict, param_file_path)

    def infer(self, document, n_steps=None, gamma=None, track_objective=False):
        """
        Get embedding.
        """
        if self.concat:
            model = "PVDMconcat"
        else:
            model = "PVDMmean"
        P_array = self.get_P_matrix()
        R_array = self.get_R_matrix()
        tokenized_doc = np.array([self.vocabulary.stoi[w] for w in document], dtype=int)
        q_vec, traj_store, obj_store = compute_embedding(
            tokenized_doc,
            R_array,
            model=model,
            P_matrix=P_array,
            mode="true",
            track_objective=track_objective,
            winsize=self.context_size,
            n_steps=n_steps,
            gamma=gamma,
        )
        return q_vec
