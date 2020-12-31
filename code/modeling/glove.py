import torch

import numpy as np
import torch.nn.functional as F
import wandb as wb

from tqdm import tqdm

from torch.utils.data import DataLoader

from .template import ContextIndependentWordVector


class GloVe(ContextIndependentWordVector):

    def __init__(self, vocab_size, hidden_dim):
        super(GloVe, self).__init__('glove')

        self._vocab_size = vocab_size
        self._hidden_dim = hidden_dim

        self._word_emb = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim,
                                            padding_idx=vocab_size - 1)
        self._context_emb = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim,
                                               padding_idx=vocab_size - 1)

        self._word_bias = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=1,
                                             padding_idx=vocab_size - 1)
        self._context_bias = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=1,
                                                padding_idx=vocab_size - 1)

    def get_embedding(self, method='word'):
        if method == 'word':
            vins = self._word_emb.weight.data
            vins = F.normalize(vins, p=2, dim=1)
            word_vectors = vins.numpy()
        elif method == 'context':
            vouts = self._context_emb.weight.data
            vouts = F.normalize(vouts, p=2, dim=1)
            word_vectors = vouts.numpy()
        elif method == 'add':
            vins = self._word_emb.weight.data
            vouts = self._context_emb.weight.data
            vadds = vins + vouts
            vadds = F.normalize(vadds, p=2, dim=1)
            word_vectors = vadds.numpy()
        elif method == 'concat':
            vins = self._word_emb.weight.data
            vouts = self._context_emb.weight.data
            vcat = torch.cat((vins, vouts), dim=1)
            vcat = F.normalize(vcat, p=2, dim=1)
            word_vectors = vcat.numpy()
        else:
            raise ValueError(f'Unrecognized method: {method}')

        return word_vectors

    def forward(self, words, contexts):
        word_v = self._word_emb(words)
        word_u = self._context_emb(contexts)

        bias_v = self._word_bias(words)
        bias_u = self._context_bias(contexts)

        dot_vu = torch.bmm(word_u, word_v.unsqueeze(-1))
        pred = ((dot_vu + bias_u).squeeze().transpose(0, 1) + bias_v.squeeze()).transpose(0, 1)

        return pred

    def train_embedding(self, corpus, opt, iterations=10_000, batch_size=64, x_max=100,
                        outpath='', save_every=1):
        word_mat = self.get_embedding('word')
        np.save(f'{outpath}word_{0:03d}.npy', word_mat)

        cntx_mat = self.get_embedding('context')
        np.save(f'{outpath}context_{0:03d}.npy', cntx_mat)

        dataloader = DataLoader(corpus, batch_size=batch_size, shuffle=True)
        for e in range(iterations):
            print(f'Begining epoch {e+1}')

            batch_losses = []
            for batch in tqdm(dataloader):
                ws = batch['word'].squeeze()
                ctxs = batch['context']
                coocs = batch['x_ij']
                weights = coocs.clone()
                weights = (weights / x_max) ** 0.75
                weights[weights > 1] = 1

                pred = self(ws, ctxs)
                log_coocs = torch.log(coocs + 1)
                diff = torch.square(pred - log_coocs)

                loss = (weights * diff).mean()

                wb.log({
                    'loss': loss.item()
                })
                batch_losses.append(loss.item())

                loss.backward()
                opt.step()
                self.zero_grad()

            print(f'Average batch loss: {np.average(batch_losses)}')

            if (e + 1) % save_every == 0:
                word_mat = self.get_embedding('word')
                np.save(f'{outpath}word_{e+1:03d}.npy', word_mat)

                cntx_mat = self.get_embedding('context')
                np.save(f'{outpath}context_{e+1:03d}.npy', cntx_mat)

