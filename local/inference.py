#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions for the inference.

"""
import time

import torch
from torch.optim import SGD
from torch import nn
from torch.nn.functional import softmax

from . import ParagraphVectorVariant

class ParagraphVectorInference(nn.Module):

    def __init__(self,
                 R_array,
                 P_array,
                 variant=ParagraphVectorVariant.PVDBOW,
                 mode="true",
                 winsize=5,
                 alpha=None):
        super().__init__()
        self.R_array = R_array.detach().cpu()
        self.P_array = P_array.detach().cpu()
        self.variant = variant
        self.mode = mode
        self.winsize = winsize
        if alpha:
            self.alpha = alpha
        else:
            self.alpha = .0001
        self.q = nn.Parameter(torch.zeros(self.R_array.shape[1]), requires_grad=True)

    def forward(self, orig):
        return compute_objective(self.q, orig, self.R_array, self.variant, self.P_array, self.mode, winsize=self.winsize, alpha=self.alpha)

    def infer(self, tokenized_doc, n_steps, gamma, track_objective=False, verbose=False):
        optimizer = SGD(params=self.parameters(), lr=gamma)
        
        infer_start = time.time()
        loss_values = []
        for i_epoch in range(n_steps):
            x = self(tokenized_doc)
            loss_values.append(x.item())
            self.zero_grad()
            x.backward()
            optimizer.step()
        infer_end = time.time()

        if verbose:
            print("auto ({:.4f}s)".format(infer_end - infer_start))

        return self.q.detach(), None, loss_values



def global_neighborhood(T, winsize=5):
    """
    All neighborhood in one array.

    PARAMETERS:
        T (int) length of the example
        winsize (int) window size, defaults to 5

    OUTPUT:
        (T-2*winsize,2*winsize) array with all neighborhood indices

    """
    aux_0 = torch.arange(winsize)
    aux_1 = torch.ones((T - 2 * winsize, 2 * winsize), dtype=torch.int)
    start = torch.cat((aux_0, winsize + 1 + aux_0))
    incr = torch.cumsum(aux_1, dim=0) - aux_1
    return start + incr


def global_context(example, winsize=5):
    """
    All contexts in one array.

    PARAMETERS:
        example (int list) encoded text
        winsize (int) window size, defaults to 5

    OUTPUT:
        (T-2*winsize,2*winsize) array with all context word indices
    """
    T = len(example)
    gn = global_neighborhood(T, winsize=winsize)
    return example[gn]


def global_context_vectors(example, P_matrix, model, winsize=5):
    """
    All context vectors for PVDM-mean.

    OUTPUT:
        (dim,T-2*winsize) array with all context vectors
    """
    gc = global_context(example, winsize=winsize)
    if model == ParagraphVectorVariant.PVDMconcat:
        T = len(example)
        D = int(P_matrix.shape[1] / (2 * winsize))
        indices = gc + D * torch.tile(torch.arange(2 * winsize), (T - 2 * winsize, 1))
        return torch.sum(P_matrix[:, indices], dim=2)
    elif model == ParagraphVectorVariant.PVDMmean:
        return torch.sum(P_matrix[:, gc], dim=2)
    else:
        print("Not implemented!")
        return 0


def global_softmax(q_vec, example, R_matrix, model=ParagraphVectorVariant.PVDBOW, P_matrix=None, winsize=5):
    """
    All the softmax values for the doc.

    OUTPUT:
        (D,T-2*winsize) array with all softmax values
    """
    T = len(example)
    if model == ParagraphVectorVariant.PVDBOW:
        aux_h = torch.tile(torch.matmul(R_matrix, q_vec), (T - 2 * winsize, 1)).T
        sm = softmax(aux_h, dim=0)
    elif model == ParagraphVectorVariant.PVDMmean or model == ParagraphVectorVariant.PVDMconcat:
        gcv = global_context_vectors(example, P_matrix, model=model, winsize=winsize)
        aux_h = (
            torch.matmul(R_matrix, gcv)
            + torch.tile(torch.matmul(R_matrix, q_vec), (T - 2 * winsize, 1)).T
        )
        sm = softmax(aux_h, dim=0)
    else:
        print("Not implemented!")
    return sm


def global_psi(q_vec, example, R_matrix, model=ParagraphVectorVariant.PVDBOW, P_matrix=None, winsize=5):
    """
    All values of \psi
    """
    T = len(example)
    aux_s = global_softmax(
        q_vec, example, R_matrix, model=model, P_matrix=P_matrix, winsize=winsize
    )
    psi_wt = -torch.log(aux_s[example[winsize : T - winsize], torch.arange(T - 2 * winsize)])
    return psi_wt


def objective_helper(
    q_vec, example, R_matrix, model=ParagraphVectorVariant.PVDBOW, P_matrix=None, winsize=5
):
    """
    Value of the objective function.
    """
    psi_wt = global_psi(
        q_vec, example, R_matrix, model=model, P_matrix=P_matrix, winsize=winsize
    )
    return torch.sum(psi_wt)


def compute_objective(
    q_vec,
    example_orig,
    R_matrix,
    model=ParagraphVectorVariant.PVDBOW,
    P_matrix=None,
    mode="true",
    example_new=None,
    lbda=None,
    pert_ind=None,
    winsize=5,
    alpha=0.0001,
):
    """
    Compute the objective function.
    """
    if mode == "true":
        obj = objective_helper(
            q_vec,
            example_orig,
            R_matrix,
            model=model,
            P_matrix=P_matrix,
            winsize=winsize,
        )
    elif mode == "linear":
        obj_orig = objective_helper(
            q_vec,
            example_orig,
            R_matrix,
            model=model,
            P_matrix=P_matrix,
            winsize=winsize,
        )
        obj_new = objective_helper(
            q_vec,
            example_new,
            R_matrix,
            model=model,
            P_matrix=P_matrix,
            winsize=winsize,
        )
        obj = lbda * obj_new + (1 - lbda) * obj_orig
    return 1./len(example_orig) * obj + 0.5 * alpha * torch.norm(q_vec) ** 2

