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

### INITIALIZE DATA & VECTORIZER

# set the seed
seed = 0
random.seed(seed)
np.random.seed(seed)

# parameters of the experiment
data = "IMDB"
model = "PVDMconcat"

# load dataset
dataset = load_dataset(data, "local", verbose=True)

# load vectorizer
vectorizer_name = get_vectorizer_name(data, "local", model)
dataset, vocabulary = dataset
variant = {
    "PVDMmean": ParagraphVectorVariant.PVDMmean,
    "PVDMconcat": ParagraphVectorVariant.PVDMconcat,
    "PVDBOW": ParagraphVectorVariant.PVDBOW,
}.get(model)
vectorizer = ParagraphVector(vocabulary, len(dataset), variant=variant)
vectorizer.load(vectorizer_name)
vocab = vectorizer.vocabulary.get_itos()

### Old implem
ex = 0
ex_orig = dataset[ex]
document = list(map(lambda u: vocab[u], ex_orig))

P_array = vectorizer.get_P_matrix()
R_array = vectorizer.get_R_matrix()
stoi = vectorizer.vocabulary.get_stoi()
tokenized_doc = torch.tensor([stoi[w] for w in document], dtype=torch.long)

manual_start = time.time()
q_vec, traj_store, obj_store = compute_embedding(
    tokenized_doc,
    R_array,
    model=vectorizer.variant,
    P_matrix=P_array,
    mode="true",
    track_objective=True,
    winsize=vectorizer.context_size,
    n_steps=1000,
    gamma=None,
)
manual_end = time.time()
print("manual ({:.4f}s)".format(manual_start - manual_end))

### New implem

q_vec_torch, _, loss_values = vectorizer.infer(document, n_steps=1000, verbose=True)

plt.semilogy(np.array(obj_store - obj_store[-1]), label="manual")
plt.semilogy(np.array(loss_values) - loss_values[-1], label="SGD")
plt.legend()
plt.show()
