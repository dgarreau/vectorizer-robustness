#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Influence of the number of replacements. Use train_model.py to train / 
calibrate the vectorizer first. 

"""

import random
import time
import pickle
import numpy as np

from gensim.models.doc2vec import Doc2Vec

from os.path import join

from utils import get_vectorizer_name
from utils import MODELS_DIR,RESULTS_DIR
from utils import load_dataset
from utils import mkdir

# fix the seed
seed = 0
random.seed(seed)
np.random.seed(seed)

# parameters of the experiment
data    = "IMDB"
implem  = "gensim"
model   = "PVDMconcat"

# get unique identifier and create relevant folder
vectorizer_name = get_vectorizer_name(data,implem,model)
res_dir = join(RESULTS_DIR,"influence_number_replacements",vectorizer_name)
mkdir(res_dir)

# load the vectorizer
f_name = join(MODELS_DIR,vectorizer_name)
if implem == 'gensim':
    vectorizer = Doc2Vec.load(f_name)
    
# load dataset
dataset = load_dataset(data,implem,verbose=True)

# get relevant parameters
winsize = vectorizer.window
dim     = vectorizer.vector_size
D       = len(vectorizer.wv)
vocab   = vectorizer.wv.index_to_key
    
# main loop
n_simu = 10
examples = [0,1,3,7,10]
for ex in examples:
    print('looking at example {}'.format(ex))
    
    # copy the current example
    ex_orig_list = dataset[ex][0].copy()
    T = len(ex_orig_list)
    
    # original embedding
    q_orig = vectorizer.infer_vector(ex_orig_list)
    
    # varying the number of replaced words
    q_new_store = np.zeros((T,n_simu,dim))
    t_start = time.time()
    for n_rep in range(1,T+1):
        
        print("{} / {}".format(n_rep,T))
        
        # Monte-Carlo loop
        for i_simu in range(n_simu):
            pert_ind = list(random.sample(range(T),n_rep))
            new_words = list(np.random.randint(0,D,size=(n_rep,)))
            ex_new_list = ex_orig_list.copy()
            
            # replace words at random
            for i_rep in range(n_rep):
                ex_new_list[pert_ind[i_rep]] = vocab[new_words[i_rep]]

            # compute the new embedding
            q_new_store[n_rep-1,i_simu,:] = vectorizer.infer_vector(ex_new_list)
    
    t_end = time.time()
    print("elapsed: {}".format(t_end-t_start))
    
    # save the results
    print("Saving results...")
    result_name = "example_" + str(ex) + ".pkl"
    file_name = join(res_dir,result_name)
    res_dict = {}
    res_dict['example_orig'] = ex_orig_list
    res_dict['q_orig']       = q_orig
    res_dict['q_new_store']  = q_new_store
    res_dict['n_rep']        = n_rep
    res_dict['n_simu']       = n_simu
    res_dict['winsize']      = winsize
    with open(file_name,'wb') as f:
        pickle.dump(res_dict,f)
    print("Done!")