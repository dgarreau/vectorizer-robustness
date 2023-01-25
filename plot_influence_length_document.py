#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Plot results for the influence of document length experiment. 

"""

import pickle
import numpy as np

import matplotlib.pyplot as plt

from os.path import join

from utils import get_vectorizer_name
from utils import RESULTS_DIR, FIGS_DIR
from utils import mkdir

# whether to save the figure or not
save_fig = True

# plot params
alpha = 0.5
small_fs = 25
large_fs = 35
lw = 3

# examples to plot
examples = [0, 1, 3]  # [0,1,3,7,10]

# parameters of the experiment
data = "IMDB"
implem = "gensim"
model = "PVDMconcat"

# get unique identifier and create relevant folders
vectorizer_name = get_vectorizer_name(data, implem, model)
res_dir = join(RESULTS_DIR, "influence_length_document", vectorizer_name)
if save_fig:
    figs_dir = join(FIGS_DIR, "influence_length_document", vectorizer_name)
    mkdir(figs_dir)

# main figure
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
for ex in examples:

    # load the results
    result_name = "example_" + str(ex) + ".pkl"
    file_name = join(res_dir, result_name)
    with open(file_name, "rb") as f:
        res_dict = pickle.load(f)
    n_length, dim = res_dict["q_orig_store"].shape
    n_simu = res_dict["q_new_store"].shape[1]

    # computing a few things before plotting
    dist_store = np.zeros((n_length, n_simu))
    for i in range(n_length):
        q_orig = res_dict["q_orig_store"][i, :]
        for i_simu in range(n_simu):
            q_new = res_dict["q_new_store"][i, i_simu]
            dist_store[i, i_simu] = np.linalg.norm(q_new - q_orig)

    mean_dist = np.mean(dist_store, axis=1)
    max_dist = np.max(dist_store, axis=1)
    min_dist = np.min(dist_store, axis=1)
    std_dist = np.std(dist_store, axis=1)

    # plotting
    t_max = len(res_dict["example_orig"])
    t_grid = np.arange(2 * res_dict["winsize"] + 1, t_max + 1)
    ax.plot(t_grid, max_dist, linewidth=lw)

    # larger tick size
    ax.tick_params(labelsize=small_fs)

# setting up the figure title and file name
# s_title = model + ", $" + str(res_dict['n_rep']) + "$ replacements"
fig_name = join(figs_dir, vectorizer_name + ".pdf")
# ax.set_title(s_title,fontsize=large_fs)

ax.set_xlabel("length of the document", fontsize=small_fs)
ax.set_ylabel(r"Euclidean distance to $q_0$", fontsize=small_fs)

if save_fig:
    fig.savefig(fig_name)
