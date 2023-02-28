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

class DirtyInference(nn.Module):

    def __init__(self,
                 R_array,
                 P_array,
                 variant=ParagraphVectorVariant.PVDBOW,
                 mode="true",):
        super().__init__()
        self.R_array = R_array
        self.P_array = P_array
        self.variant = variant
        self.mode = mode
        self.q = nn.Parameter(torch.zeros(self.R_array.shape[1]), requires_grad=True)

    def forward(self, orig):
        return compute_objective(self.q, orig, self.R_array, self.variant, self.P_array, self.mode, winsize=vectorizer.context_size)


model= DirtyInference(R_array, P_array, variant)

# optimizer
optimizer = SGD(params=model.parameters(), lr=0.01)
n_epochs = 1000

# entering the main loop
auto_start = time.time()
loss_values = []
for i_epoch in range(n_epochs):
    x = model(ex_orig)
    loss_values.append(x.item())
    model.zero_grad()
    x.backward()
    optimizer.step()
auto_end = time.time()

print("auto ({:.4f}s)".format(auto_start - auto_end))

plt.semilogy(np.array(obj_store - obj_store[-1]), label="manual")
plt.semilogy(np.array(loss_values) - loss_values[-1], label="SGD")
plt.legend()
plt.show()
