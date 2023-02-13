#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Utils.

Kudos to https://github.com/inejc/paragraph-vectors for the data interface and
https://stackoverflow.com/questions/11373610/save-matplotlib-file-to-a-directory
for mkdir.

"""

import re
import random
import urllib.request
import numpy as np
import gensim
import sys
import smart_open
import pandas as pd
from collections import Counter

from os.path import join
from os import makedirs, path
import configparser

from errno import EEXIST

from torch.utils.data import Subset

# from torchtext.legacy.data import Field, TabularDataset # TODO: to remove after newimplem
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab

# set the seed
seed = 0
random.seed(seed)
np.random.seed(seed)

# getting local config
config = configparser.ConfigParser()
config.read("config.ini")
_root_dir = path.expanduser(config.get("files", "root_path"))

DATA_DIR = join(_root_dir, "data")
FIGS_DIR = join(_root_dir, "figures")
MODELS_DIR = join(_root_dir, "models")
RESULTS_DIR = join(_root_dir, "results")


def get_vectorizer_name(data, implem, model):
    return data + "_" + implem + "_" + model


def mkdir(mypath):
    """
    Creates a directory.
    """
    try:
        makedirs(mypath)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else:
            raise


def load_dataset(data, implem, verbose=False, split_ratio=None):
    # taking only 1000 doc
    if not split_ratio:
        split_ratio = 0.02

    if verbose:
        print("loading data...")

    if data == "IMDB":

        n_docs = 50000
        file_path = join(DATA_DIR, "IMDB-Dataset.csv")
        if not path.exists(file_path):
            download_IMDB_dataset()

        if implem == "new_local":
            # FIXME: randomize and include train also
            dataset = IMDB(DATA_DIR, split="train")
            #dataset = Subset(dataset, range(int(split_ratio * n_docs)))
            tokenizer = get_tokenizer('basic_english')

            n_train = int(split_ratio * n_docs)
            counter = Counter()
            cur_n = 0
            for (label, line) in dataset:
                if cur_n < n_train:
                    counter.update(tokenizer(line))
                else:
                    break
            vocabulary = vocab(counter, min_freq=1)

            dataset = {
                "dataset": dataset,
                "vocab": vocabulary,
                "counter": counter,
            }

        elif implem == "local":

            text_field = Field(pad_token=None, tokenize=_tokenize_str)
            class_field = Field()
            initial = TabularDataset(
                path=file_path,
                format="csv",
                fields=[("text", text_field), ("label", class_field)],
                skip_header=True,
            )

            dataset, _ = initial.split(
                split_ratio=split_ratio, random_state=random.getstate()
            )

            if verbose:
                print("building vocabulary...")
            text_field.build_vocab(dataset)
            if verbose:
                print("done!")
        elif implem == "gensim":

            dataset = list(read_corpus(file_path, int(split_ratio * n_docs)))

        elif implem == "scikit":

            df = pd.read_csv(file_path)
            dataset = list(df["review"][: int(split_ratio * n_docs)])

        else:
            print("not implemented")

    else:
        dataset = []
        print("not implemented")

    print("done!")
    print()
    return dataset


def read_corpus(fname, n_docs=None, tokens_only=False):
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
    str_ = re.sub(r"[^A-Za-z0-9(),.!?\'`]", " ", str_)
    # str_ = re.sub(r'[^A-Za-z0-9]', ' ', str_)

    # remove multiple whitespace characters
    str_ = re.sub(r"\s{2,}", " ", str_)

    # punctations to tokens
    str_ = re.sub(r"\(", " ( ", str_)
    str_ = re.sub(r"\)", " ) ", str_)
    str_ = re.sub(r",", " , ", str_)
    str_ = re.sub(r"\.", " . ", str_)
    str_ = re.sub(r"!", " ! ", str_)
    str_ = re.sub(r"\?", " ? ", str_)

    # split contractions into multiple tokens
    str_ = re.sub(r"\'s", " 's", str_)
    str_ = re.sub(r"\'ve", " 've", str_)
    str_ = re.sub(r"n\'t", " n't", str_)
    str_ = re.sub(r"\'re", " 're", str_)
    str_ = re.sub(r"\'d", " 'd", str_)
    str_ = re.sub(r"\'ll", " 'll", str_)

    # lower case
    return str_.strip().lower().split()


def download_IMDB_dataset():
    # Download IMDB dataset into data folder
    # TODO: Would be better to directly download from Kaggle (but require account)
    url = "https://github.com/Ankit152/IMDB-sentiment-analysis/blob/master/IMDB-Dataset.csv?raw=true"
    filename = "IMDB-Dataset.csv"
    filepath = path.join(DATA_DIR, filename)
    if not path.exists(filepath):
        mkdir(DATA_DIR)
        print("Downloading IMDB dataset...")
        urllib.request.urlretrieve(url, filepath)
        print("Done!")
    else:
        print("IMDB dataset already downloaded")
