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
                 winsize=5):
        super().__init__()
        self.R_array = R_array
        self.P_array = P_array
        self.variant = variant
        self.mode = mode
        self.winsize = winsize
        self.q = nn.Parameter(torch.zeros(self.R_array.shape[1]), requires_grad=True)

    def forward(self, orig):
        return compute_objective(self.q, orig, self.R_array, self.variant, self.P_array, self.mode, winsize=self.winsize)

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
    return obj


def global_indic(example, D, winsize=5):
    """
    All indicator functions (\indic{\word_t} in our notation)

    OUTPUT
        (D,T-2*winsize) array
    """
    T = len(example)
    aux_1 = torch.zeros((D, T - 2 * winsize))
    aux_1[example[winsize : T - winsize], torch.arange(T - 2 * winsize)] = 1.0
    return aux_1


def global_gradient(q_vec, example, R_matrix, model=ParagraphVectorVariant.PVDBOW, P_matrix=None, winsize=5):
    """
    All gradients.
    """
    D, dim = R_matrix.shape
    aux_s = global_softmax(
        q_vec, example, R_matrix, model=model, P_matrix=P_matrix, winsize=winsize
    )
    aux_1 = global_indic(example, D, winsize=winsize)
    return torch.matmul(R_matrix.T, aux_s - aux_1)


def gradient_helper(q_vec, example, R_matrix, model=ParagraphVectorVariant.PVDBOW, P_matrix=None, winsize=5):
    """
    Return the full gradient at the point.
    """
    all_grad = global_gradient(
        q_vec, example, R_matrix, model=model, P_matrix=P_matrix, winsize=winsize
    )
    return torch.sum(all_grad, dim=1)

# TODO: Transform it in Torch code
def compute_gradient(
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
):
    """
    Compute the full gradient for the different models.
    """
    D, dim = R_matrix.shape
    T = len(example_orig)

    if model == ParagraphVectorVariant.PVDBOW:
        if mode == "true":
            grad = (T - 2 * winsize) * torch.matmul(
                R_matrix.T, softmax(torch.matmul(R_matrix, q_vec), dim=0)
            ) - torch.sum(R_matrix[example_orig[winsize : T - winsize], :], 0)

        elif mode == "linear":
            grad_orig = (T - 2 * winsize) * torch.matmul(
                R_matrix.T, softmax(torch.matmul(R_matrix, q_vec), dim=0)
            ) - torch.sum(R_matrix[example_orig[winsize : T - winsize], :], 0)
            grad_new = (T - 2 * winsize) * torch.matmul(
                R_matrix.T, softmax(torch.matmul(R_matrix, q_vec), dim=0)
            ) - torch.sum(R_matrix[example_new[winsize : T - winsize], :], 0)
            grad = lbda * grad_new + (1 - lbda) * grad_orig

        else:
            print("not implemented!")
    elif model == ParagraphVectorVariant.PVDMmean or model == ParagraphVectorVariant.PVDMconcat:
        if mode == "true":
            grad = gradient_helper(
                q_vec,
                example_orig,
                R_matrix,
                model=model,
                P_matrix=P_matrix,
                winsize=winsize,
            )
        elif mode == "linear":
            grad_orig = gradient_helper(
                q_vec,
                example_orig,
                R_matrix,
                model=model,
                P_matrix=P_matrix,
                winsize=winsize,
            )
            grad_new = gradient_helper(
                q_vec,
                example_new,
                R_matrix,
                model=model,
                P_matrix=P_matrix,
                winsize=winsize,
            )
            grad = lbda * grad_new + (1 - lbda) * grad_orig
        else:
            print("Not implemented!")
    else:
        print("Not implemented!")
        grad = torch.zeros((dim,))
    return grad


def compute_embedding(
    example_orig,
    R_matrix,
    model=ParagraphVectorVariant.PVDBOW,
    P_matrix=None,
    mode="true",
    example_new=None,
    lbda=None,
    pert_ind=None,
    winsize=5,
    gamma=None,
    n_steps=None,
    track_objective=False,
    verbose=False,
):
    """
    Computing the embedding.
    """
    if gamma is None:
        if model == ParagraphVectorVariant.PVDMmean or model == ParagraphVectorVariant.PVDMconcat:
            gamma = 0.01
        else:
            gamma = 0.001
    if n_steps is None:
        if model == ParagraphVectorVariant.PVDMmean or model == ParagraphVectorVariant.PVDMconcat:
            n_steps = 100
        else:
            n_steps = 200
    # HACK: make it work on CPU for the moment
    R_matrix = R_matrix.cpu()
    P_matrix = P_matrix.cpu()
    D, dim = R_matrix.shape
    q_vec = torch.zeros((dim,))
    traj_store = torch.zeros((n_steps, dim))
    traj_store[0] = q_vec
    obj_store = torch.zeros((n_steps,))
    if track_objective:
        obj_store[0] = compute_objective(
            q_vec,
            example_orig,
            R_matrix,
            model=model,
            P_matrix=P_matrix,
            mode=mode,
            example_new=example_new,
            lbda=lbda,
            pert_ind=pert_ind,
            winsize=winsize,
        )
    # main optim loop
    for step in range(1, n_steps):
        grad_t = compute_gradient(
            q_vec,
            example_orig,
            R_matrix,
            model=model,
            P_matrix=P_matrix,
            mode=mode,
            example_new=example_new,
            lbda=lbda,
            pert_ind=pert_ind,
            winsize=winsize,
        )
        q_vec = q_vec - gamma * grad_t
        traj_store[step] = q_vec
        if track_objective:
            obj_store[step] = compute_objective(
                q_vec,
                example_orig,
                R_matrix,
                model=model,
                P_matrix=P_matrix,
                mode=mode,
                example_new=example_new,
                lbda=lbda,
                pert_ind=pert_ind,
                winsize=winsize,
            )
    return q_vec, traj_store, obj_store
