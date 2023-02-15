from sys import stdout
import time

import numpy as np

import torch
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
            print("N = {:d}".format(model.n_docs))
            print("D = {:d}".format(model.n_words))
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
