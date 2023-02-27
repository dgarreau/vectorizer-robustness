import random
import numpy as np
import matplotlib.pyplot as plt
import torch

from local.inference import compute_embedding
from utils import load_dataset, get_vectorizer_name
from local.models import ParagraphVector, ParagraphVectorVariant

### INITIALIZE DATA & VECTORIZER

# set the seed
seed = 0
random.seed(seed)
np.random.seed(seed)

# parameters of the experiment
data = "IMDB"
model = "PVDBOW"

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
q_vec, traj_store, obj_store = compute_embedding(
    tokenized_doc,
    R_array,
    model=vectorizer.variant,
    P_matrix=P_array,
    mode="true",
    track_objective=True,
    winsize=vectorizer.context_size,
    n_steps=100,
    gamma=None,
)

### New implem

