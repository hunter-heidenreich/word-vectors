import torch

import numpy as np

from .template import ContextIndependentWordVector


def get_negative_samples(corpus, size, k=20):
    sz = (size[0], size[1], k)

    samples = torch.LongTensor(corpus.sample_tokens(sz))
    samples = samples.view(sz[0], -1)

    return samples


class Skipgram(ContextIndependentWordVector):
    def __init__(self, vocab_size, hidden_dim=64):
        super(Skipgram, self).__init__('word2vec_SG')

        self._vocab_size = vocab_size
        self._hidden_dim = hidden_dim

        self.vectors_in = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim,
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


class CBOW(ContextIndependentWordVector):

    def __init__(self, vocab_size, hidden_dim=64):
        super(CBOW, self).__init__('word2vec_CBOW')

        self._vocab_size = vocab_size
        self._hidden_dim = hidden_dim

        self.vectors_in = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim,
                                             padding_idx=vocab_size-1)
        self.vectors_out = torch.nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        initrange = 0.5 / self._hidden_dim
        self.vectors_in.weight.data.uniform_(-initrange, initrange)
        self.vectors_out.weight.data.uniform_(-0, 0)

    def forward(self, contexts, mask=None):
        # Index into the input matrix
        v_ins = self.vectors_in(contexts)

        # Average context vectors
        # (masking allows different sized contexts to be averaged correctly)
        if mask is not None:
            v_in = torch.zeros((v_ins.size(0), self._hidden_dim))
            v_ins_sum = v_ins.sum(dim=1)
            for dim in range(v_ins_sum.size(0)):
                v_in[dim] = v_ins_sum[dim] / mask[dim].sum()
        else:
            v_in = v_ins.mean(dim=1)

        # Output prediction vectors (logits)
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

            cnt, ctx, msk = gen_func(corpus, batch_size, window=rand_m)
            pred = self(ctx, mask=msk)

            loss = loss_func(pred, cnt)

            loss.backward()
            opt.step()
            self.zero_grad()

            if step % 10 == 0:
                print(f'Step {step}: {loss:.4f} loss')


class SkipgramNS(ContextIndependentWordVector):

    def __init__(self, vocab_size, hidden_dim=64):
        super(SkipgramNS, self).__init__('word2vec_SGNS')

        self._vocab_size = vocab_size
        self._hidden_dim = hidden_dim

        self.V_centers = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim)
        self.U_contexts = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim)

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

    def get_embedding(self):
        vs = self.V_centers.weight.data.numpy()
        us = self.U_contexts.weight.data.numpy()

        word_vectors = np.concatenate((vs, us), axis=-1)

        return word_vectors

    def train_embedding(self, corpus, gen_func, opt, loss_func,
                        iterations=10_000, batch_size=64, window_len=10,
                        neg_samples=20):
        for step in range(iterations):
            # As discussed in https://arxiv.org/pdf/1402.3722.pdf
            # m is a maximal context window,
            # therefore we'll randomly draw a number
            # between 1 and m, using that random value
            # as the realized context window size
            rand_m = np.random.randint(1, window_len)
            cnt, ctx, _ = gen_func(corpus, batch_size, window=rand_m)
            neg = get_negative_samples(corpus, ctx.shape, k=neg_samples)

            pos_pred, neg_pred = self(cnt, ctx, neg)
            pos_gold = torch.ones(pos_pred.shape)
            neg_gold = torch.zeros(neg_pred.shape)

            loss = loss_func(pos_pred, pos_gold) + loss_func(neg_pred, neg_gold)
            loss /= batch_size * ctx.size(1)

            loss.backward()
            opt.step()
            self.zero_grad()

            if step % 10 == 0:
                print(f'Step {step}: {loss:.4f} loss')


class CBOWNS(ContextIndependentWordVector):

    def __init__(self, vocab_size, hidden_dim=64):
        super(CBOWNS, self).__init__('word2vec_CBOWNS')

        self._vocab_size = vocab_size
        self._hidden_dim = hidden_dim

        self.V_centers = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim)
        self.U_contexts = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim)

        self._init_weights()

    def _init_weights(self):
        initrange = 0.5 / self._hidden_dim
        self.V_centers.weight.data.uniform_(-initrange, initrange)
        self.U_contexts.weight.data.uniform_(-initrange, initrange)

    def forward(self, centers, contexts, neg_centers, mask=None):
        # Index into the input matrix
        context_us = self.U_contexts(contexts)

        # Average context vectors
        # (masking allows different sized contexts to be averaged correctly)
        if mask is not None:
            context_hs = torch.zeros((context_us.size(0), self._hidden_dim))
            context_us_sum = context_us.sum(dim=1)
            for dim in range(context_us_sum.size(0)):
                context_hs[dim] = context_us_sum[dim] / mask[dim].sum()
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

    def get_embedding(self):
        vs = self.V_centers.weight.data.numpy()
        us = self.U_contexts.weight.data.numpy()

        word_vectors = np.concatenate((vs, us), axis=-1)

        return word_vectors

    def train_embedding(self, corpus, gen_func, opt, loss_func,
                        iterations=10_000, batch_size=64, window_len=10,
                        neg_samples=20):
        for step in range(iterations):
            # As discussed in https://arxiv.org/pdf/1402.3722.pdf
            # m is a maximal context window,
            # therefore we'll randomly draw a number
            # between 1 and m, using that random value
            # as the realized context window size
            rand_m = np.random.randint(1, window_len)

            cnt, ctx, msk = gen_func(corpus, batch_size, window=rand_m)
            neg = get_negative_samples(corpus, ctx.shape, k=neg_samples)
            pos_pred, neg_pred = self(cnt, ctx, neg, mask=msk)

            pos_gold = torch.ones(pos_pred.shape)
            neg_gold = torch.zeros(neg_pred.shape)

            loss = loss_func(pos_pred, pos_gold) + loss_func(neg_pred, neg_gold)
            loss /= batch_size * ctx.size(1)

            loss.backward()
            opt.step()
            self.zero_grad()

            if step % 10 == 0:
                print(f'Step {step}: {loss:.4f} loss')
