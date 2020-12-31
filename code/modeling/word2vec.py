import torch

import numpy as np
import torch.nn.functional as F
import wandb as wb

from tqdm import tqdm

from torch.utils.data import DataLoader

from .template import ContextIndependentWordVector


class Skipgram(ContextIndependentWordVector):
    def __init__(self, vocab_size, hidden_dim=64):
        super(Skipgram, self).__init__('word2vec_SG')

        self._vocab_size = vocab_size
        self._hidden_dim = hidden_dim

        self.vectors_in = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim,
                                             padding_idx=vocab_size - 1)
        self.vectors_out = torch.nn.Linear(hidden_dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        initrange = 0.5 / self._hidden_dim
        self.vectors_in.weight.data.uniform_(-initrange, initrange)

    def forward(self, centers):
        v_in = self.vectors_in(centers)
        v_out = self.vectors_out(v_in)
        return v_out

    def get_embedding(self, method='word'):
        if method == 'word':
            vins = self.vectors_in.weight.data
            vins = F.normalize(vins, p=2, dim=1)
            word_vectors = vins.numpy()
        elif method == 'context':
            vouts = self.vectors_out.weight.data
            vouts = F.normalize(vouts, p=2, dim=1)
            word_vectors = vouts.numpy()
        elif method == 'add':
            vins = self.vectors_in.weight.data
            vouts = self.vectors_out.weight.data
            vadds = vins + vouts
            vadds = F.normalize(vadds, p=2, dim=1)
            word_vectors = vadds.numpy()
        elif method == 'concat':
            vins = self.vectors_in.weight.data
            vouts = self.vectors_out.weight.data
            vcat = torch.cat((vins, vouts), dim=1)
            vcat = F.normalize(vcat, p=2, dim=1)
            word_vectors = vcat.numpy()
        else:
            raise ValueError(f'Unrecognized method: {method}')

        return word_vectors

    def train_embedding(self, corpus, opt, loss_func, iterations=10_000, batch_size=64, outpath='', save_every=1):
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

                ctx_pred = self(ws)
                ctx_pred = ctx_pred.unsqueeze(-1)
                ctx_pred = ctx_pred.repeat(1, 1, ctxs.size(-1))
                ctx_pred = ctx_pred.transpose(1, 2)
                ctx_pred = ctx_pred.reshape((ctx_pred.size(0) * ctx_pred.size(1), ctx_pred.size(2)))

                ctx_gold = ctxs.clone()
                ctx_gold = ctx_gold.flatten()
                ctx_gold[ctx_gold == corpus.lookup(corpus._pad_tok)] = -100

                loss = loss_func(ctx_pred, ctx_gold)

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


class CBOW(ContextIndependentWordVector):

    def __init__(self, vocab_size, hidden_dim=64):
        super(CBOW, self).__init__('word2vec_CBOW')

        self._vocab_size = vocab_size
        self._hidden_dim = hidden_dim

        self.vectors_in = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim,
                                             padding_idx=vocab_size-1)
        self.vectors_out = torch.nn.Linear(hidden_dim, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        initrange = 0.5 / self._hidden_dim
        self.vectors_in.weight.data.uniform_(-initrange, initrange)

    def forward(self, contexts, weight=None):
        # Index into the input matrix
        v_ins = self.vectors_in(contexts)

        if weight is not None:
            weight = weight.unsqueeze(1)
            v_in = torch.bmm(weight, v_ins).squeeze()
        else:
            v_in = v_ins.mean(dim=1)

        # Output prediction vectors (logits)
        v_out = self.vectors_out(v_in)

        return v_out

    def get_embedding(self, method='word'):
        if method == 'word':
            vins = self.vectors_in.weight.data
            vins = F.normalize(vins, p=2, dim=1)
            word_vectors = vins.numpy()
        elif method == 'context':
            vouts = self.vectors_out.weight.data
            vouts = F.normalize(vouts, p=2, dim=1)
            word_vectors = vouts.numpy()
        elif method == 'add':
            vins = self.vectors_in.weight.data
            vouts = self.vectors_out.weight.data
            vadds = vins + vouts
            vadds = F.normalize(vadds, p=2, dim=1)
            word_vectors = vadds.numpy()
        elif method == 'concat':
            vins = self.vectors_in.weight.data
            vouts = self.vectors_out.weight.data
            vcat = torch.cat((vins, vouts), dim=1)
            vcat = F.normalize(vcat, p=2, dim=1)
            word_vectors = vcat.numpy()
        else:
            raise ValueError(f'Unrecognized method: {method}')

        return word_vectors

    def train_embedding(self, corpus, opt, loss_func, iterations=10_000, batch_size=64, outpath='', save_every=1):
        dataloader = DataLoader(corpus, batch_size=batch_size, shuffle=True)

        word_mat = self.get_embedding('word')
        np.save(f'{outpath}word_{0:03d}.npy', word_mat)

        cntx_mat = self.get_embedding('context')
        np.save(f'{outpath}context_{0:03d}.npy', cntx_mat)

        for e in range(iterations):
            print(f'Begining epoch {e+1}')

            batch_losses = []
            for batch in tqdm(dataloader):
                ws = batch['word'].squeeze()
                ctxs = batch['context']
                weight = batch['weight']

                ws_pred = self(ctxs, weight=weight)
                ws_gold = ws.clone()

                loss = loss_func(ws_pred, ws_gold)

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


class SkipgramNS(ContextIndependentWordVector):

    def __init__(self, vocab_size, hidden_dim=64):
        super(SkipgramNS, self).__init__('word2vec_SGNS')

        self._vocab_size = vocab_size
        self._hidden_dim = hidden_dim

        self.V_centers = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim, padding_idx=vocab_size - 1)
        self.U_contexts = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim, padding_idx=vocab_size - 1)

        self._init_weights()

    def _init_weights(self):
        initrange = 0.5 / self._hidden_dim
        self.V_centers.weight.data.uniform_(-initrange, initrange)
        self.U_contexts.weight.data.uniform_(-initrange, initrange)

    def forward(self, centers, contexts, neg_contexts):
        center_vs = self.V_centers(centers)  # batch_size X hidden_dim
        center_vs = center_vs.unsqueeze(1)  # batch_size X 1 X hidden_dim

        context_us = self.U_contexts(contexts)  # batch_size X pos_examples X hidden_dim
        context_us = context_us.transpose(1, 2)  # batch_size X hidden_dim X pos_examples

        pos_scores = torch.bmm(center_vs, context_us)  # batch_size X 1 X pos_examples
        pos_scores = pos_scores.squeeze()  # batch_sie X pos_examples
        pos_scores = pos_scores.view(-1)

        context_ns = self.U_contexts(neg_contexts)  # batch_size X neg_examples X hidden_dim
        context_ns = context_ns.transpose(1, 2)  # batch_size X hidden_dim X neg_examples

        neg_scores = torch.bmm(center_vs, context_ns)  # batch_size X 1 X neg_examples
        neg_scores = -1 * neg_scores.squeeze()  # batch_sie X neg_examples
        neg_scores = neg_scores.view(-1)

        return pos_scores, neg_scores

    def get_embedding(self, method='word'):
        if method == 'word':
            vins = self.V_centers.weight.data
            vins = F.normalize(vins, p=2, dim=1)
            word_vectors = vins.numpy()
        elif method == 'context':
            vouts = self.U_contexts.weight.data
            vouts = F.normalize(vouts, p=2, dim=1)
            word_vectors = vouts.numpy()
        elif method == 'add':
            vins = self.V_centers.weight.data
            vouts = self.U_contexts.weight.data
            vadds = vins + vouts
            vadds = F.normalize(vadds, p=2, dim=1)
            word_vectors = vadds.numpy()
        elif method == 'concat':
            vins = self.V_centers.weight.data
            vouts = self.U_contexts.weight.data
            vcat = torch.cat((vins, vouts), dim=1)
            vcat = F.normalize(vcat, p=2, dim=1)
            word_vectors = vcat.numpy()
        else:
            raise ValueError(f'Unrecognized method: {method}')

        return word_vectors

    def train_embedding(self, corpus, opt, loss_func, iterations=10_000, batch_size=64, neg_samples=20,
                        outpath='', save_every=1):

        word_mat = self.get_embedding('word')
        np.save(f'{outpath}word_{0:03d}.npy', word_mat)

        cntx_mat = self.get_embedding('context')
        np.save(f'{outpath}context_{0:03d}.npy', cntx_mat)

        dataloader = DataLoader(corpus, batch_size=batch_size, shuffle=True)
        for e in range(iterations):
            print(f'Begining epoch {e + 1}')

            batch_losses = []
            for batch in tqdm(dataloader):
                ws = batch['word'].squeeze()
                ctxs = batch['context']
                negs = corpus.sample_tokens((ws.size(0), neg_samples))

                pos_pred, neg_pred = self(ws, ctxs, negs)
                pos_gold = torch.ones(pos_pred.shape)
                neg_gold = torch.zeros(neg_pred.shape)

                pos_loss = loss_func(pos_pred, pos_gold) / ws.size(0)
                neg_loss = loss_func(neg_pred, neg_gold) / ws.size(0)
                loss = pos_loss + neg_loss

                wb.log({
                    'loss':     loss.item(),
                    'pos_loss': pos_loss.item(),
                    'neg_loss': neg_loss.item()
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


class CBOWNS(ContextIndependentWordVector):

    def __init__(self, vocab_size, hidden_dim=64):
        super(CBOWNS, self).__init__('word2vec_CBOWNS')

        self._vocab_size = vocab_size
        self._hidden_dim = hidden_dim

        self.V_centers = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim,
                                            padding_idx=vocab_size - 1)
        self.U_contexts = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim,
                                             padding_idx=vocab_size - 1)

        self._init_weights()

    def _init_weights(self):
        initrange = 0.5 / self._hidden_dim
        self.V_centers.weight.data.uniform_(-initrange, initrange)
        self.U_contexts.weight.data.uniform_(-initrange, initrange)

    def forward(self, centers, contexts, neg_centers, weight=None):
        # Index into the input matrix
        context_us = self.U_contexts(contexts)

        # Average context vectors
        if weight is not None:
            weight = weight.unsqueeze(1)
            context_hs = torch.bmm(weight, context_us).squeeze()
        else:
            context_hs = context_us.mean(dim=1)  # batch_size X hidden_dim
        context_hs = context_hs.unsqueeze(1)  # batch_size X 1 X hidden_dim

        center_vs = self.V_centers(centers)  # batch_size X hidden_dim
        center_vs = center_vs.unsqueeze(-1)  # batch_size X hidden_dim X 1

        pos_scores = torch.bmm(context_hs, center_vs)  # batch_size X 1 X 1
        pos_scores = pos_scores.squeeze()

        center_ns = self.V_centers(neg_centers)  # batch_size X neg_examples X hidden_dim
        center_ns = center_ns.transpose(1, 2)  # batch_size X hidden_dim X neg_examples

        neg_scores = torch.bmm(context_hs, center_ns)  # batch_size X 1 X neg_examples
        neg_scores = -1 * neg_scores.squeeze()  # batch_sie X neg_examples
        neg_scores = neg_scores.view(-1)

        return pos_scores, neg_scores

    def get_embedding(self, method='word'):
        if method == 'word':
            vins = self.V_centers.weight.data
            vins = F.normalize(vins, p=2, dim=1)
            word_vectors = vins.numpy()
        elif method == 'context':
            vouts = self.U_contexts.weight.data
            vouts = F.normalize(vouts, p=2, dim=1)
            word_vectors = vouts.numpy()
        elif method == 'add':
            vins = self.V_centers.weight.data
            vouts = self.U_contexts.weight.data
            vadds = vins + vouts
            vadds = F.normalize(vadds, p=2, dim=1)
            word_vectors = vadds.numpy()
        elif method == 'concat':
            vins = self.V_centers.weight.data
            vouts = self.U_contexts.weight.data
            vcat = torch.cat((vins, vouts), dim=1)
            vcat = F.normalize(vcat, p=2, dim=1)
            word_vectors = vcat.numpy()
        else:
            raise ValueError(f'Unrecognized method: {method}')

        return word_vectors

    def train_embedding(self, corpus, opt, loss_func, iterations=10_000, batch_size=64, neg_samples=20,
                        outpath='', save_every=1):
        dataloader = DataLoader(corpus, batch_size=batch_size, shuffle=True)

        word_mat = self.get_embedding('word')
        np.save(f'{outpath}word_{0:03d}.npy', word_mat)

        cntx_mat = self.get_embedding('context')
        np.save(f'{outpath}context_{0:03d}.npy', cntx_mat)

        for e in range(iterations):
            print(f'Begining epoch {e + 1}')

            batch_losses = []
            for batch in tqdm(dataloader):
                ws = batch['word'].squeeze()
                ctxs = batch['context']
                weight = batch['weight']
                negs = corpus.sample_tokens((ws.size(0), neg_samples))

                pos_pred, neg_pred = self(ws, ctxs, negs, weight=weight)
                pos_gold = torch.ones(pos_pred.shape)
                neg_gold = torch.zeros(neg_pred.shape)

                pos_loss = loss_func(pos_pred, pos_gold) / ws.size(0)
                neg_loss = loss_func(neg_pred, neg_gold) / ws.size(0)
                loss = pos_loss + neg_loss

                wb.log({
                    'loss':     loss.item(),
                    'pos_loss': pos_loss.item(),
                    'neg_loss': neg_loss.item()
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
