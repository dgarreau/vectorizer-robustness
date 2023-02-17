#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Re-implementation of PVDM and PVDBOW models. 

kudos to https://github.com/inejc/paragraph-vectors

"""

# TODO (carefuly) implement 1/T

import torch
import torch.nn as nn

from utils import MODELS_DIR

from os.path import join

from .inference import compute_embedding
from . import ParagraphVectorVariant

#FIXME I don't like the fact that __init__ depends on the number of documents.
class ParagraphVector(nn.Module):
    def __init__(self, vocabulary, n_docs, context_size=5, dim=50, variant=ParagraphVectorVariant.PVDBOW):
        super(ParagraphVector, self).__init__()
        self.vocabulary = vocabulary
        self.dim = dim
        self.context_size = context_size
        self.n_words = len(self.vocabulary)
        self.n_docs = n_docs
        self.variant = variant

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
            self._P_matrix = nn.Parameter(torch.zeros((self.n_words, self.dim))) #to be able to save it
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

    def extra_repr(self):
        return f"dim={self.dim}, context_size={self.context_size}, n_words={self.n_words}, n_docs={self.n_docs},"

    def get_P_matrix(self):
        return self._P_matrix.T.detach()

    def get_Q_matrix(self):
        return self._Q_matrix.T.detach()

    def get_R_matrix(self):
        return self._R_matrix.T.detach()

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
        stoi = self.vocabulary.get_stoi()
        tokenized_doc = torch.tensor([stoi[w] for w in document], dtype=torch.long)
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
        return q_vec#, traj_store, obj_store
