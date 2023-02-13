#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Functions for the inference.

"""

import numpy as np

from scipy.special import softmax

from numpy.matlib import repmat


def global_neighborhood(T, winsize=5):
    """
    All neighborhood in one array.

    PARAMETERS:
        T (int) length of the example
        winsize (int) window size, defaults to 5

    OUTPUT:
        (T-2*winsize,2*winsize) array with all neighborhood indices

    """
    aux_0 = np.arange(winsize)
    aux_1 = np.ones((T - 2 * winsize, 2 * winsize), dtype=int)
    start = np.concatenate((aux_0, winsize + 1 + aux_0))
    incr = np.cumsum(aux_1, axis=0) - aux_1
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
    if model == "PVDMconcat":
        T = len(example)
        D = int(P_matrix.shape[1] / (2 * winsize))
        indices = gc + D * repmat(np.arange(2 * winsize), T - 2 * winsize, 1)
        return np.sum(P_matrix[:, indices], axis=2)
    elif model == "PVDMmean":
        return np.sum(P_matrix[:, gc], axis=2)
    else:
        print("Not implemented!")
        return 0


def global_softmax(q_vec, example, R_matrix, model="PVDBOW", P_matrix=None, winsize=5):
    """
    All the softmax values for the doc.

    OUTPUT:
        (D,T-2*winsize) array with all softmax values
    """
    T = len(example)
    if model == "PVDBOW":
        aux_h = repmat(np.dot(R_matrix, q_vec), T - 2 * winsize, 1).T
        sm = softmax(aux_h, axis=0)
    elif model == "PVDMmean" or model == "PVDMconcat":
        gcv = global_context_vectors(example, P_matrix, model=model, winsize=winsize)
        aux_h = (
            np.dot(R_matrix, gcv)
            + repmat(np.dot(R_matrix, q_vec), T - 2 * winsize, 1).T
        )
        sm = softmax(aux_h, axis=0)
    else:
        print("Not implemented!")
    return sm


def global_psi(q_vec, example, R_matrix, model="PVDBOW", P_matrix=None, winsize=5):
    """
    All values of \psi
    """
    T = len(example)
    aux_s = global_softmax(
        q_vec, example, R_matrix, model=model, P_matrix=P_matrix, winsize=winsize
    )
    psi_wt = -np.log(aux_s[example[winsize : T - winsize], np.arange(T - 2 * winsize)])
    return psi_wt


def objective_helper(
    q_vec, example, R_matrix, model="PVDBOW", P_matrix=None, winsize=5
):
    """
    Value of the objective function.
    """
    psi_wt = global_psi(
        q_vec, example, R_matrix, model=model, P_matrix=P_matrix, winsize=winsize
    )
    return np.sum(psi_wt)


def compute_objective(
    q_vec,
    example_orig,
    R_matrix,
    model="PVDBOW",
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
    aux_1 = np.zeros((D, T - 2 * winsize))
    aux_1[example[winsize : T - winsize], np.arange(T - 2 * winsize)] = 1.0
    return aux_1


def global_gradient(q_vec, example, R_matrix, model="PVDBOW", P_matrix=None, winsize=5):
    """
    All gradients.
    """
    D, dim = R_matrix.shape
    aux_s = global_softmax(
        q_vec, example, R_matrix, model=model, P_matrix=P_matrix, winsize=winsize
    )
    aux_1 = global_indic(example, D, winsize=winsize)
    return np.dot(R_matrix.T, aux_s - aux_1)


def gradient_helper(q_vec, example, R_matrix, model="PVDBOW", P_matrix=None, winsize=5):
    """
    Return the full gradient at the point.
    """
    all_grad = global_gradient(
        q_vec, example, R_matrix, model=model, P_matrix=P_matrix, winsize=winsize
    )
    return np.sum(all_grad, axis=1)

# TODO: Transform it in Torch code
def compute_gradient(
    q_vec,
    example_orig,
    R_matrix,
    model="PVDBOW",
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

    if model == "PVDBOW":
        if mode == "true":
            grad = (T - 2 * winsize) * np.dot(
                R_matrix.T, softmax(np.dot(R_matrix, q_vec))
            ) - np.sum(R_matrix[example_orig[winsize : T - winsize], :], 0)

        elif mode == "linear":
            grad_orig = (T - 2 * winsize) * np.dot(
                R_matrix.T, softmax(np.dot(R_matrix, q_vec))
            ) - np.sum(R_matrix[example_orig[winsize : T - winsize], :], 0)
            grad_new = (T - 2 * winsize) * np.dot(
                R_matrix.T, softmax(np.dot(R_matrix, q_vec))
            ) - np.sum(R_matrix[example_new[winsize : T - winsize], :], 0)
            grad = lbda * grad_new + (1 - lbda) * grad_orig

        else:
            print("not implemented!")
    elif model == "PVDMmean" or model == "PVDMconcat":
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
        grad = np.zeros((dim,))
    return grad


def compute_embedding(
    example_orig,
    R_matrix,
    model="PVDBOW",
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
        if model == "PVDMmean" or model == "PVDMconcat":
            gamma = 0.01
        else:
            gamma = 0.005
    if n_steps is None:
        if model == "PVDMmean" or model == "PVDMconcat":
            n_steps = 100
        else:
            n_steps = 200
    D, dim = R_matrix.shape
    q_vec = np.zeros((dim,))
    traj_store = np.zeros((n_steps, dim))
    traj_store[0] = q_vec
    obj_store = np.zeros((n_steps,))
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
