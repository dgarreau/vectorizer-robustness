#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Training the models.

WARNING: hs=0 and negative=0 does not what you think it does and results in 
garbarge results, see https://groups.google.com/g/gensim/c/TCIrgMagoFc

"""

import random
import numpy as np
import gensim
import time
import pickle

from os.path import join

from local.models import PVDM,PVDBOW

from sklearn.feature_extraction.text import TfidfVectorizer
from utils import get_vectorizer_name
from utils import load_dataset
from utils import MODELS_DIR

# fix the seed
seed = 0
random.seed(seed)
np.random.seed(seed)

# parameters of the experiment
data   = "IMDB"
implem = "local"# scikit, gensim, local
model  = "PVDMmean"# TFIDF, PVDMmean, PVDMconcat, PVDBOW

# unique identifier
vectorizer_name = get_vectorizer_name(data,implem,model)

# load data
dataset = load_dataset(data,implem,verbose=True)

# instanciate the model
winsize = 5
if implem == 'gensim':
    
    dim = 50
    n_epochs = 100
    
    
    if model == 'PVDMmean':
        dm = 1
        dm_mean   = 1
        dm_concat = 0
        dbow_words = 1
    elif model == 'PVDMconcat':
        dm = 1
        dm_mean   = 0
        dm_concat = 1
        dbow_words = 1
    elif model == 'PVDBOW':
        dm = 0
        dm_mean = None
        dm_concat = None
        dbow_words = 0
        
    vectorizer = gensim.models.doc2vec.Doc2Vec(vector_size=dim, 
                                      min_count=2, 
                                      epochs=n_epochs,
                                      dm=dm,
                                      dm_mean=dm_mean,
                                      dbow_words=dbow_words,
                                      negative=0,
                                      hs=1,
                                      window=winsize)

    # build the voc 
    vectorizer.build_vocab(dataset)
    
    # train the model
    print("Training the model...")
    t_start = time.time()
    vectorizer.train(dataset, total_examples=vectorizer.corpus_count, epochs=vectorizer.epochs)
    t_end = time.time()
    print("Done!")
    print("Elapsed: {}s".format(np.round(t_end-t_start,2)))
    print()

    # save the model to disk
    f_name = join(MODELS_DIR,vectorizer_name)
    vectorizer.save(f_name)

elif implem == 'scikit':
    
    if model == 'TFIDF':
        
        # default = l2 normalization
        vectorizer = TfidfVectorizer()
        vectorizer.fit(dataset)
    
        f_name = join(MODELS_DIR,vectorizer_name)
        with open(f_name, 'wb') as f:
            pickle.dump(vectorizer,f)
    
    else: 
        print('not implemented')

elif implem == 'local':
    
    dim = 50
    winsize = 5
    n_epochs = 10
    
    if model == 'PVDMmean':
        lr = 0.001 
        vectorizer = PVDM(dim=dim,context_size=winsize,concat=False)
    elif model == 'PVDMconcat':
        lr = 0.0005
        vectorizer = PVDM(dim=dim,context_size=winsize,concat=True)
    elif model == 'PVDBOW':
        lr = 0.001
        PVDBOW(dim=dim,context_size=winsize)

    # training the model
    vectorizer.train(dataset,lr=lr,n_epochs=n_epochs,verbose=True)
    
    # saving the model
    vectorizer.save()

else:
    
    print('not implemented')


