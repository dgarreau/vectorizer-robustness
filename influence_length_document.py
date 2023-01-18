#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Script for the influence of document length experiments. Use train_model.py to
train the vectorizer first. 

"""

import random
import time
import pickle
import numpy as np

from local.models import PVDM,PVDBOW

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
implem  = "local"
model   = "PVDMmean"

# get unique identifier and create relevant folder
vectorizer_name = get_vectorizer_name(data,implem,model)
res_dir = join(RESULTS_DIR,"influence_length_document",vectorizer_name)
mkdir(res_dir)

# load the vectorizer
f_name = join(MODELS_DIR,vectorizer_name)
if implem == 'gensim':
    vectorizer = Doc2Vec.load(f_name)
elif implem == 'scikit':
    with open(f_name,'rb') as f:
        vectorizer = pickle.load(f)
elif implem == 'local':
    if model == 'PVDMmean':
        vectorizer = PVDM(concat=False)
    elif model == 'PVDMconcat':
        vectorizer = PVDM(concat=True)
    elif model == 'PVDBOW':
        PVDBOW()
    vectorizer.load(vectorizer_name)

# load dataset
dataset = load_dataset(data,implem,verbose=True)

# get relevant parameters
if implem == 'gensim':
    winsize = vectorizer.window
    dim     = vectorizer.vector_size
    D       = len(vectorizer.wv)
    vocab   = vectorizer.wv.index_to_key
elif implem == 'scikit':
    # NOTE: winsize is arbitrary here
    winsize = 5
    vocab = vectorizer.get_feature_names_out()
    D = len(vocab)
    dim = D
elif implem == 'local':
    winsize = vectorizer.get_context_size()
    dim = vectorizer.get_dim()
    D = vectorizer.get_n_words()
    vocab = vectorizer.vocabulary
    
# main loop
n_rep = 5
n_simu = 50
examples = [0,1,3,7,10]
for ex in examples:
    print('looking at example {}'.format(ex))
    
    # copy the current example
    if implem == 'gensim':
        ex_orig_list = dataset[ex][0].copy()
    elif implem == 'scikit':
        ex_orig = dataset[ex]
        ex_orig_list = ex_orig.split(' ')
    
    # range of the experiment
    t_max = len(ex_orig_list)
    n_length = t_max - 2*winsize
    
    # where the results are stored
    q_orig_store = np.zeros((n_length,dim))
    q_new_store = np.zeros((n_length,n_simu,dim))

    # example loop starts
    t_start = time.time()
    for current_length in range(2*winsize+1,t_max+1):
        print("{} / {}".format(current_length-2*winsize,n_length))
        ex_current_list = ex_orig_list[:current_length].copy()
        ex_current = ' '.join(ex_current_list)
        
        if implem == 'gensim':
            q_orig_store[current_length-2*winsize-1] = vectorizer.infer_vector(ex_current_list)
        elif implem == 'scikit':
            q_orig_store[current_length-2*winsize-1] = vectorizer.transform([ex_current]).todense()
#        elif implem == 'local':
#            q_orig_store[current_length-2*winsize-1] = 
            
        # Monte-Carlo loop starts
        for i_simu in range(n_simu):
            pert_ind = list(random.sample(range(current_length),n_rep))
            new_words = list(np.random.randint(0,D,size=(n_rep,)))
            ex_new_list = ex_current_list.copy()
            
            # replace words at random
            for i_rep in range(n_rep):
                ex_new_list[pert_ind[i_rep]] = vocab[new_words[i_rep]]
            ex_new = ' '.join(ex_new_list)
            
            # get vectorization of the new document
            if implem == 'gensim':
                q_new_store[current_length-2*winsize-1,i_simu,:] = vectorizer.infer_vector(ex_new_list)
            elif implem == 'scikit':
                q_new_store[current_length-2*winsize-1,i_simu,:] = vectorizer.transform([ex_new]).todense()
                
    t_end = time.time()
    print("elapsed: {}".format(t_end-t_start))
    
    # save the results
    print("Saving results...")
    result_name = "example_" + str(ex) + ".pkl"
    file_name = join(res_dir,result_name)
    res_dict = {}
    res_dict['example_orig'] = ex_orig_list
    res_dict['q_orig_store'] = q_orig_store
    res_dict['q_new_store']  = q_new_store
    res_dict['n_rep']        = n_rep
    res_dict['n_simu']       = n_simu
    res_dict['winsize']      = winsize
    with open(file_name,'wb') as f:
        pickle.dump(res_dict,f)
    print("Done!")




