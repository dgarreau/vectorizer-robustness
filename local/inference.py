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
                 context_size=5,
                 alpha=None):
        super().__init__()
        self.R_array = R_array.detach().cpu()
        self.P_array = P_array.detach().cpu()
        self.variant = variant
        self.mode = mode
        self.context_size = context_size
        if alpha:
            self.alpha = alpha
        else:
            self.alpha = .0001
        self.q = nn.Parameter(torch.zeros(self.R_array.shape[1]), requires_grad=True)

    def forward(self, orig):
        return compute_objective(self.q, orig, self.R_array, self.variant, self.P_array, self.mode, context_size=self.context_size, alpha=self.alpha)

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



def global_neighborhood(T, context_size=5):
    """
    All neighborhood in one array.

    PARAMETERS:
        T (int) length of the example
        context_size (int) window size, defaults to 5

    OUTPUT:
        (T-2*context_size,2*context_size) array with all neighborhood indices

    """
    aux_0 = torch.arange(context_size)
    aux_1 = torch.ones((T - 2 * context_size, 2 * context_size), dtype=torch.int)
    start = torch.cat((aux_0, context_size + 1 + aux_0))
    incr = torch.cumsum(aux_1, dim=0) - aux_1
    return start + incr


def global_context(example, context_size=5):
    """
    All contexts in one array.

    PARAMETERS:
        example (int list) encoded text
        context_size (int) window size, defaults to 5

    OUTPUT:
        (T-2*context_size,2*context_size) array with all context word indices
    """
    T = len(example)
    gn = global_neighborhood(T, context_size=context_size)
    return example[gn]


def global_context_vectors(example, P_matrix, model, context_size=5):
    """
    All context vectors for PVDM-mean.

    OUTPUT:
        (dim,T-2*context_size) array with all context vectors
    """
    gc = global_context(example, context_size=context_size)
    if model == ParagraphVectorVariant.PVDMconcat:
        T = len(example)
        D = int(P_matrix.shape[1] / (2 * context_size))
        indices = gc + D * torch.tile(torch.arange(2 * context_size), (T - 2 * context_size, 1))
        return torch.sum(P_matrix[:, indices], dim=2)
    elif model == ParagraphVectorVariant.PVDMmean:
        return torch.sum(P_matrix[:, gc], dim=2)
    else:
        raise NotImplementedError


def global_softmax(q_vec, example, R_matrix, model=ParagraphVectorVariant.PVDBOW, P_matrix=None, context_size=5):
    """
    All the softmax values for the doc.

    OUTPUT:
        (D,T-2*context_size) array with all softmax values
    """
    T = len(example)
    if model == ParagraphVectorVariant.PVDBOW:
        aux_h = torch.tile(torch.matmul(R_matrix, q_vec), (T - 2 * context_size, 1)).T
        sm = softmax(aux_h, dim=0)
    elif model == ParagraphVectorVariant.PVDMmean or model == ParagraphVectorVariant.PVDMconcat:
        gcv = global_context_vectors(example, P_matrix, model=model, context_size=context_size)
        aux_h = (
            torch.matmul(R_matrix, gcv)
            + torch.tile(torch.matmul(R_matrix, q_vec), (T - 2 * context_size, 1)).T
        )
        sm = softmax(aux_h, dim=0)
    else:
        raise NotImplementedError
    return sm


def global_psi(q_vec, example, R_matrix, model=ParagraphVectorVariant.PVDBOW, P_matrix=None, context_size=5):
    """
    All values of \psi
    """
    T = len(example)
    aux_s = global_softmax(
        q_vec, example, R_matrix, model=model, P_matrix=P_matrix, context_size=context_size
    )
    psi_wt = -torch.log(aux_s[example[context_size : T - context_size], torch.arange(T - 2 * context_size)])
    return psi_wt


def objective_helper(
    q_vec, example, R_matrix, model=ParagraphVectorVariant.PVDBOW, P_matrix=None, context_size=5
):
    """
    Value of the objective function.
    """
    psi_wt = global_psi(
        q_vec, example, R_matrix, model=model, P_matrix=P_matrix, context_size=context_size
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
    context_size=5,
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
            context_size=context_size,
        )
    elif mode == "linear":
        obj_orig = objective_helper(
            q_vec,
            example_orig,
            R_matrix,
            model=model,
            P_matrix=P_matrix,
            context_size=context_size,
        )
        obj_new = objective_helper(
            q_vec,
            example_new,
            R_matrix,
            model=model,
            P_matrix=P_matrix,
            context_size=context_size,
        )
        obj = lbda * obj_new + (1 - lbda) * obj_orig
    return 1./len(example_orig) * obj + 0.5 * alpha * torch.norm(q_vec) ** 2

