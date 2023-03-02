import random
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam, SGD


from utils import load_dataset, get_vectorizer_name
from local.models import ParagraphVector, ParagraphVectorVariant
from local.inference import *

if __name__ == "__main__":

    ### INITIALIZE DATA & VECTORIZER

    # set the seed
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    # parameters of the experiment
    data = "IMDB"
    model = ParagraphVectorVariant.PVDBOW

    # load dataset
    dataset = load_dataset(data, "local", verbose=True)

    # load vectorizer
    vectorizer_name = get_vectorizer_name(data, "local", model)
    dataset, vocabulary = dataset
    vectorizer = ParagraphVector(vocabulary, len(dataset), variant=model)
    vectorizer.load(vectorizer_name)
    vocab = vectorizer.vocabulary.get_itos()

    ex = 0
    ex_orig = dataset[ex]
    document = list(map(lambda u: vocab[u], ex_orig))
    q_vec_torch, _, loss_values = vectorizer.infer(
        document, n_steps=100, verbose=True, track_objective=True
    )
    q_vec_direct = vectorizer.new_infer(dataset[:100], n_epochs=100)

    plt.semilogy(np.array(loss_values) - loss_values[-1], label="SGD")
    plt.legend()
    plt.show()
