#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Looking at the norm of the original document as a function of T for doc2vec.

"""

import numpy as np
import random
import time
import matplotlib.pyplot as plt

from os.path import join

from gensim.models.doc2vec import Doc2Vec

from utils import get_vectorizer_name
from utils import MODELS_DIR,RESULTS_DIR,FIGS_DIR
from utils import mkdir
from utils import load_dataset

# set the seed
seed = 0
random.seed(seed)
np.random.seed(seed)

# parameters of the experiment
data    = "IMDB"
implem  = "gensim"
model   = "PVDBOW"

winsize = 5

# get unique identifier and create relevant folder
vectorizer_name = get_vectorizer_name(data,implem,model)
res_dir = join(RESULTS_DIR,"influence_length_document",vectorizer_name)
mkdir(res_dir)

figs_dir = join(FIGS_DIR,"norm_original_embedding",vectorizer_name)
mkdir(figs_dir)

# load the vectorizer
f_name = join(MODELS_DIR,vectorizer_name)
vectorizer = Doc2Vec.load(f_name)
dim     = vectorizer.vector_size

# load dataset
dataset = load_dataset(data,implem,verbose=True)
N = len(dataset)

# find the example with the max length
length_store = np.zeros((N,))
for i in range(N):
    length_store[i] = len(dataset[i][0])
    
# number of examples to consider
k = 5
examples = np.argpartition(length_store,-k)[-k:]

norm_dict = {}
for ex in examples:
    
    print('looking at example {}'.format(ex))
    T = len(dataset[ex][0])
    print('length {}'.format(T))

    ex_orig_list = dataset[ex][0].copy()

    # range of the experiment
    t_max = len(ex_orig_list)
    n_length = t_max - 2*winsize
    
    q_orig_store = np.zeros((n_length,dim))
    
    t_start = time.time()
    for current_length in range(2*winsize+1,t_max+1):
        print("{} / {}".format(current_length-2*winsize,n_length))
        ex_current_list = ex_orig_list[:current_length].copy()
        ex_current = ' '.join(ex_current_list)
        q_orig_store[current_length-2*winsize-1] = vectorizer.infer_vector(ex_current_list)
    t_end = time.time()
    print("elapsed: {}".format(t_end-t_start))   

    norm_dict[ex] = np.linalg.norm(q_orig_store,axis=1)

##############################################################################

# plot params
alpha = 0.5
small_fs = 25
large_fs = 35
lw = 3

fig,ax = plt.subplots(1,1,figsize=(10,8))

for ex in examples:
    ax.plot(norm_dict[ex])

# setting up the figure title and file name
s_title = model + ", norm of $q_0$"
fig_name = join(figs_dir,vectorizer_name + ".pdf")
ax.set_title(s_title,fontsize=large_fs)

ax.set_xlabel("length of the document",fontsize=small_fs)
ax.set_ylabel(r"norm of $q_0$",fontsize=small_fs)

ax.tick_params(labelsize=small_fs)

fig.savefig(fig_name)

