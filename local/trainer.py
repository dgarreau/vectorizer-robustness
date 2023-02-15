from sys import stdout
import time

import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .data import ContexifiedDataSet
from .loss import LogSoftmax
from . import ParagraphVectorVariant

def _print_progress(epoch_i, batch_i, num_batches):
    progress = round((batch_i + 1) / num_batches * 100)
    print("\rEpoch {:d}".format(epoch_i + 1), end="")
    stdout.write(" - {:d}%".format(progress))
    stdout.flush()


class Trainer:
    def __init__(self,
                 lr=0.002,
                 n_epochs=100,
                 batch_size=32,
                 num_workers=8,
                 verbose=False):
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.verbose = verbose


    def fit(self, model, dataset):
        raw_data, _ = dataset
        ctx_dataset = ContexifiedDataSet(
            raw_data,
            model.context_size,
        )
        dataloader = DataLoader(
            ctx_dataset, self.batch_size, num_workers=self.num_workers
        )
        n_batches = len(dataloader)

        self.n_docs = len(raw_data)
        self.n_words = len(model.vocabulary)

        # Projection matrix P
        if model.variant == ParagraphVectorVariant.PVDMconcat:
            # size 2nuD x d in our notation
            model._P_matrix = nn.Parameter(
                torch.sqrt(
                    torch.tensor([2.0])
                    / torch.tensor([2 * model.context_size * self.n_words + model.dim])
                )
                * torch.randn(2 * model.context_size * self.n_words, model.dim),
                requires_grad=True,
            )
            model.one_hot_buffer = nn.Parameter(
                torch.eye(self.n_words), requires_grad=False
            )
        elif model.variant == ParagraphVectorVariant.PVDMmean:
            # size d x D in our notation
            model._P_matrix = nn.Parameter(
                torch.sqrt(
                    torch.tensor([2.0]) / torch.tensor([self.n_words + model.dim])
                )
                * torch.randn(self.n_words, model.dim),
                requires_grad=True,
            )
        elif model.variant == ParagraphVectorVariant.PVDBOW:
            model._P_matrix = nn.Parameter(torch.tensor([])) #to be able to save it
        else:
            raise NotImplementedError

        # Matrix R
        # final layer
        # R in our notation, size D x d
        model._R_matrix = nn.Parameter(
            torch.sqrt(torch.tensor([2.0]) / torch.tensor([self.n_words + model.dim]))
            * torch.randn(model.dim, self.n_words),
            requires_grad=True,
        )

        # embedding of the documents, Gaussian initialization
        # Q in our notation, size d x N
        model._Q_matrix = nn.Parameter(
            torch.sqrt(torch.tensor([2.0]) / torch.tensor([self.n_docs + model.dim]))
            * torch.randn(self.n_docs, model.dim),
            requires_grad=True,
        )

        if torch.cuda.is_available():
            model.cuda()
            if self.verbose:
                print("using cuda")
                print()

        if self.verbose:
            if model.variant == ParagraphVectorVariant.PVDMconcat:
                print("PVDM-concat training starts:")
            elif model.variant == ParagraphVectorVariant.PVDMmean:
                print("PVDM-mean training starts:")
            elif model.variant == ParagraphVectorVariant.PVDBOW:
                print("PVDBOW training starts:")
            else:
                raise NotImplementedError
            print("N = {:d}".format(self.n_docs))
            print("D = {:d}".format(self.n_words))
            print("d = {:d}".format(model.dim))
            print("nu = {:d}".format(model.context_size))
            print("lr = {:.4f}".format(self.lr))
            print()

        # loss function
        cost_func = LogSoftmax()

        # optimizer
        optimizer = Adam(params=model.parameters(), lr=self.lr)

        # entering the main loop
        t_start = time.time()
        loss_values = []
        for i_epoch in range(self.n_epochs):
            epoch_start = time.time()
            loss = []
            i_batch = 0  # enumerate is bad for multiprocessing?
            for batch in dataloader:
                context_ids, doc_ids, target_ids = batch
                x = model(context_ids, doc_ids)
                x = cost_func(x, target_ids)
                loss.append(x.item())
                model.zero_grad()
                x.backward()
                optimizer.step()
                if self.verbose:
                    _print_progress(i_epoch, i_batch, n_batches)
                i_batch += 1

            # end of epoch
            loss = torch.mean(torch.FloatTensor(loss))
            loss_values.append(loss)

            epoch_end = time.time()

            epoch_total_time = round(epoch_end - epoch_start)
            if self.verbose:
                print(" ({:d}s) - loss: {:.4f}".format(epoch_total_time, loss))
        t_end = time.time()

        # saving relevant info
        self.loss_values = loss_values
        if self.verbose:
            print("elapsed = {}s".format(np.round(t_end - t_start, 0)))
