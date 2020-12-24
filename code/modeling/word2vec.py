import torch

import numpy as np

from .template import ContextIndependentWordVector


class Skipgram(ContextIndependentWordVector):
    def __init__(self, vocab_size, hidden_dim=64):
        super(Skipgram, self).__init__('word2vec_SG')

        self._vocab_size = vocab_size
        self._hidden_dim = hidden_dim

        self.vectors_in = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim, sparse=True,
                                             padding_idx=vocab_size - 1)
        self.vectors_out = torch.nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        initrange = 0.5 / self._hidden_dim
        self.vectors_in.weight.data.uniform_(-initrange, initrange)
        self.vectors_out.weight.data.uniform_(-0, 0)

    def forward(self, centers):
        v_in = self.vectors_in(centers)
        v_out = self.vectors_out(v_in)
        return v_out

    def get_embedding(self):
        vins = self.vectors_in.weight.data.numpy()
        vouts = self.vectors_out.weight.data.numpy()

        word_vectors = np.concatenate((vins, vouts), axis=-1)

        return word_vectors

    def train_embedding(self, corpus, gen_func, opt, loss_func,
                        iterations=10_000, batch_size=64, window_len=10):
        for step in range(iterations):
            # As discussed in https://arxiv.org/pdf/1402.3722.pdf
            # m is a maximal context window,
            # therefore we'll randomly draw a number
            # between 1 and m, using that random value
            # as the realized context window size
            rand_m = np.random.randint(1, window_len)

            cnt, ctx, _ = gen_func(corpus, batch_size, window=rand_m)

            pred = self(cnt)

            # Expand prediction instead of running through multiple times
            pred = pred.unsqueeze(-1)
            pred = pred.repeat(1, 1, ctx.size(-1))
            pred = pred.transpose(1, 2)
            pred = pred.reshape((pred.size(0) * pred.size(1), pred.size(2)))

            ctx = ctx.flatten()
            ctx[ctx == corpus.lookup(corpus._pad_tok)] = -100

            loss = loss_func(pred, ctx)

            loss.backward()
            opt.step()
            self.zero_grad()

            if step % 10 == 0:
                print(f'Step {step}: {loss:.4f} loss')
