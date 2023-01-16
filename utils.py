#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Utils.

Credits to https://github.com/inejc/paragraph-vectors for the datainterface.

"""

import re
import random
import numpy as np
import gensim
import sys
import smart_open

from os.path import join
from os import makedirs
from os import path

from errno import EEXIST

from torchtext.legacy.data import Field, TabularDataset

# fix the seed
seed = 0
random.seed(seed)
np.random.seed(seed)

# change here for local use
_root_dir = "/home/dgarreau/Documents/research_local/current_projects/vectorizer/"

DATA_DIR    = join(_root_dir, 'data')
FIGS_DIR    = join(_root_dir, 'figures')
MODELS_DIR  = join(_root_dir,'models')
RESULTS_DIR = join(_root_dir, 'results')

def get_vectorizer_name(data,implem,model):
    return data + "_" + implem + "_" + model

def mkdir(mypath):
    """
    Creates a directory (credits to https://stackoverflow.com/questions/11373610/save-matplotlib-file-to-a-directory)
    
    INPUT:
        - mypath: str with path to the directory    
    """
    try:
        makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: 
            raise

def load_dataset(data,implem,verbose=False,split_ratio=0.1):

    if verbose:
        print('loading data...')
            
    if data == 'IMDB':
        
        n_docs = 50000
        file_path = join(DATA_DIR,'IMDB-Dataset.csv')
        
        if implem == 'local':

            text_field = Field(pad_token=None, tokenize=_tokenize_str)
            class_field = Field()
            initial = TabularDataset(
                    path=file_path,
                    format='csv',
                    fields=[('text', text_field),('label',class_field)],
                    skip_header=True)
            
            dataset,_ = initial.split(split_ratio=split_ratio,random_state=random.getstate())
            
            if verbose:
                print('building vocabulary...')
            text_field.build_vocab(dataset)
            if verbose:
                print('done!')
        elif implem == 'gensim':

            dataset = list(read_corpus(file_path,int(split_ratio*n_docs)))
        
        else:
            print('not implemented')
        
    else:
        dataset = []
        print('not implemented')
    
    print('done!')
    print()
    return dataset

def read_corpus(fname,n_docs=None,tokens_only=False):
    """
    Read corpus from file for gensim.
    """
    if n_docs is None:
        n_docs = sys.float_info.max
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if i > 0 and i < n_docs:
                tokens = gensim.utils.simple_preprocess(line)
                if tokens_only:
                    yield tokens
                else:
                    # For training data, add tags
                    yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

def _tokenize_str(str_):
    
    # keep only alphanumeric and punctuation signs
    str_ = re.sub(r'[^A-Za-z0-9(),.!?\'`]', ' ', str_)
    #str_ = re.sub(r'[^A-Za-z0-9]', ' ', str_)
    
    # remove multiple whitespace characters
    str_ = re.sub(r'\s{2,}', ' ', str_)
    
    # punctations to tokens
    str_ = re.sub(r'\(', ' ( ', str_)
    str_ = re.sub(r'\)', ' ) ', str_)
    str_ = re.sub(r',', ' , ', str_)
    str_ = re.sub(r'\.', ' . ', str_)
    str_ = re.sub(r'!', ' ! ', str_)
    str_ = re.sub(r'\?', ' ? ', str_)
    
    # split contractions into multiple tokens
    str_ = re.sub(r'\'s', ' \'s', str_)
    str_ = re.sub(r'\'ve', ' \'ve', str_)
    str_ = re.sub(r'n\'t', ' n\'t', str_)
    str_ = re.sub(r'\'re', ' \'re', str_)
    str_ = re.sub(r'\'d', ' \'d', str_)
    str_ = re.sub(r'\'ll', ' \'ll', str_)
    
    # lower case
    return str_.strip().lower().split()




