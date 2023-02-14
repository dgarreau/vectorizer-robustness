#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Training / calibrating the models. Only dataset available is IMDB.

Implementations are:
 - scikit: using scikit-learn implmentation (for TFIDF)
 - local: local implementation of PV
 - gensim: external implementation of PV
 
WARNING: local implementation is very slow, even on GPU. Moreover, step size 
needs to be tuned. 

Models are:
 - TFIDF
 - PVDMmean
 - PVDMconcat
 - PVDBOW

WARNING: with the gensim implementation, taking hs=0 and negative=0 does not do
what you think it does and may result in strange results, see 
https://groups.google.com/g/gensim/c/TCIrgMagoFc

"""

import random
import numpy as np
import gensim
import time
import pickle

from os.path import join

from local.models import ParagraphVector, ParagraphVectorVariant

from sklearn.feature_extraction.text import TfidfVectorizer
from utils import get_vectorizer_name
from utils import load_dataset
from utils import MODELS_DIR, mkdir

if __name__ == '__main__':

    # create folder to save data
    mkdir(MODELS_DIR)

    # setthe seed
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    # parameters of the experiment
    data = "IMDB"
    implem = "local"
    model = "PVDBOW"

    # unique identifier
    vectorizer_name = get_vectorizer_name(data, implem, model)

    # load data
    dataset = load_dataset(data, implem, split_ratio = 0.002, verbose=True)

    # instanciate the model
    winsize = 5
    if implem == "gensim":

        dim = 50
        n_epochs = 100

        if model == "PVDMmean":
            dm = 1
            dm_mean = 1
            dm_concat = 0
            dbow_words = 1
        elif model == "PVDMconcat":
            dm = 1
            dm_mean = 0
            dm_concat = 1
            dbow_words = 1
        elif model == "PVDBOW":
            dm = 0
            dm_mean = None
            dm_concat = None
            dbow_words = 0

        vectorizer = gensim.models.doc2vec.Doc2Vec(
            vector_size=dim,
            min_count=2,
            epochs=n_epochs,
            dm=dm,
            dm_mean=dm_mean,
            dbow_words=dbow_words,
            negative=0,
            hs=1,
            window=winsize,
        )

        # build the voc
        vectorizer.build_vocab(dataset)

        # train the model
        print("Training the model...")
        t_start = time.time()
        vectorizer.train(
            dataset, total_examples=vectorizer.corpus_count, epochs=vectorizer.epochs
        )
        t_end = time.time()
        print("Done!")
        print("Elapsed: {}s".format(np.round(t_end - t_start, 2)))
        print()

        # save the model to disk
        f_name = join(MODELS_DIR, vectorizer_name)
        mkdir(MODELS_DIR)
        vectorizer.save(f_name)

    elif implem == "scikit":

        if model == "TFIDF":

            # default = l2 normalization
            vectorizer = TfidfVectorizer()
            vectorizer.fit(dataset)

            mkdir(MODELS_DIR)
            f_name = join(MODELS_DIR, vectorizer_name)
            with open(f_name, "wb") as f:
                pickle.dump(vectorizer, f)

        else:
            print("not implemented")

    elif implem == "local":

        dim = 50
        winsize = 5
        n_epochs = 2#100

        if model == "PVDMmean":
            lr = 0.001
            vectorizer = ParagraphVector(dim=dim, context_size=winsize, variant=ParagraphVectorVariant.PVDMmean)
        elif model == "PVDMconcat":
            lr = 0.0005
            vectorizer = ParagraphVector(dim=dim, context_size=winsize, variant=ParagraphVectorVariant.PVDMconcat)
        elif model == "PVDBOW":
            lr = 0.001
            vectorizer = ParagraphVector(dim=dim, context_size=winsize, variant=ParagraphVectorVariant.PVDBOW)

        # training the model
        vectorizer.train(dataset, lr=lr, n_epochs=n_epochs, verbose=True)

        # saving the model
        vectorizer.save(vectorizer_name, verbose=True)

    else:

        print("not implemented")
