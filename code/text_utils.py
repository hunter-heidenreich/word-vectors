import re
import torch

import numpy as np

from collections import Counter, defaultdict
from glob import glob
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from torch.utils.data import Dataset


def tokenize(text, space=False, lower=False):
    text = re.sub(r"([^\n])\n([^\n])", r"\1 \2", text)  # collapse in-paragraph newline formatting
    text = re.sub("[ ]+", " ", text)  # collapse whitespace padding
    text = re.sub('--', ' ', text)  # turn '--' into a space (to separate tokens)
    tokens = []
    for token in re.split("([0-9a-zA-Z\'\-]+)", text):
        if re.search("[0-9a-zA-Z\'\-]", token):
            tokens.append(token.lower() if lower else token)
        elif space:
            tokens.extend(token.lower() if lower else token)
    return tokens


class Corpus(Dataset):

    DATA_ROOT = 'data/'

    def __init__(self, path,
                 min_freq=-1,  # minimum number of times for a word type to be retained
                 max_vocab=-1,  # maximum size of vocabulary
                 sub=-1,  # sub-sampling threshold
                 m=5,  # context window size
                 alpha=1,  # smoothing parameter for sampling from the smoothed unigram distribution
                 lower=True,
                 dirty=True,  # if true, will drop words prior to slicing context windows (otherwise after)
                 dyn=False,  # if true, window is maximal (and randomly sampled each generation)
                 start_token='<s>',
                 end_token='</s>',
                 pad_token='<pad>'
                 ):
        self._path = path

        self._min_freq = min_freq
        self._max_vocab = max_vocab
        self._sub = sub
        self._m = m
        self._alpha = alpha

        self._lower = lower
        self._dirty = dirty
        self._dyn = dyn

        self._start_tok = start_token
        self._end_tok = end_token
        self._pad_tok = pad_token

        self._corpus = []
        self._counts = {}
        self._word_ctx = defaultdict(list)
        self._word_co = defaultdict(lambda: defaultdict(int))

        self._token2idx = {}
        self._idx2token = {}

        self._idx2prob = []

        self._pairs = []

        self._load()

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, item):
        return self._pairs[item]

    def _load(self):
        print('Reading raw data')
        for f in glob(self.DATA_ROOT + self._path):
            with open(f) as fp:
                for paragraph in re.split("\n\n+", fp.read().strip()):
                    paragraph = [self._start_tok] + \
                                [word
                                    for sentence in sent_tokenize(paragraph)
                                    for word in tokenize(sentence.strip(), lower=self._lower)
                                    if sentence.strip()] + \
                                [self._end_tok]

                    if len(paragraph) > 2:
                        self._corpus.append(paragraph)
        print(f'Loaded {len(self._corpus)} paragraphs')

        print('Computing token counts')
        self._counts = dict(Counter([w for s in tqdm(self._corpus) for w in s]))
        train_tokens = sum(self._counts.values())
        print(f'Loaded {len(self._counts)} token types; {train_tokens} token instances')

        # cache for comparison
        original_size = len(self._counts)
        original_train_tokens = train_tokens

        if not self._dirty:
            self._slice_corpus()

        if self._min_freq > 0:
            print(f'Dropping words with a frequency < {self._min_freq}')

            self._drop_rare()

            new_size = len(self._counts)
            train_tokens = sum(self._counts.values())
            print(f'Vocab is now {new_size} token types ({100 * new_size / original_size:.0f}% of original types)')
            print(f'Corpus is now {train_tokens} tokens ({100 * train_tokens / original_train_tokens:.0f}% of original corpus)')

        if self._sub:
            print(f'Sub-sampling words at a occurrence threshold of {self._sub}')

            self._sub_sample()

            new_size = len(self._counts)
            train_tokens = sum(self._counts.values())
            print(f'Vocab is now {new_size} token types ({100 * new_size / original_size:.0f}% of original types)')
            print(f'Corpus is now {train_tokens} tokens ({100 * train_tokens / original_train_tokens:.0f}% of original corpus)')

        if 0 < self._max_vocab < len(self._counts):
            print(f'Reducing vocabulary to top {self._max_vocab} token types')

            self._truncate_vocab()

            new_size = len(self._counts)
            train_tokens = sum(self._counts.values())
            print(f'Vocab is now {new_size} token types ({100 * new_size / original_size:.0f}% of original types)')
            print(f'Corpus is now {train_tokens} tokens ({100 * train_tokens / original_train_tokens:.0f}% of original corpus)')

        if self._dirty:
            self._slice_corpus()

        print('Computing token lookup')
        self._token2idx = {
            t: idx for idx, (t, _) in enumerate(sorted(self._counts.items(), reverse=True, key=lambda kv: kv[1]))
        }
        self._token2idx[self._pad_tok] = len(self._token2idx)
        self._idx2token = {v: k for k, v in self._token2idx.items()}

        print('Constructing sample distribution')
        base = 0
        for i in tqdm(range(self.vocab_size)):
            tok = self._idx2token[i]

            if tok in self._counts:
                val = np.power(self._counts[tok], self._alpha)
                base += val
                self._idx2prob.append(val)
            else:
                self._idx2prob.append(0)

        self._idx2prob = [(v / base) for v in self._idx2prob]

        print(f'Encoding pairs')
        self._generate_pairs()

    def _slice_corpus(self):
        self._word_ctx = defaultdict(list)
        self._word_co = defaultdict(list)
        for p in tqdm(self._corpus):
            p_len = len(p)
            for ix, tok in enumerate(p):
                lctx = p[max(0, ix-self._m):ix]
                rctx = p[min(ix+1, p_len-1):min(ix+1+self._m, p_len-1)]

                # pad context for even-batch processing
                ctx = [self._pad_tok] * (self._m - len(lctx)) + lctx + rctx + [self._pad_tok] * (self._m - len(rctx))
                self._word_ctx[tok].append(ctx)
                self._word_co[tok].extend(lctx + rctx)

        self._word_co = {word: Counter(toks) for word, toks in self._word_co.items()}

    def _drop_rare(self):
        if self._dirty:
            self._counts = {w: cnt for w, cnt in self._counts.items() if cnt > self._min_freq}
            self._corpus = [[w for w in s if w in self._counts] for s in tqdm(self._corpus)]
            self._corpus = [s for s in tqdm(self._corpus) if len(s) > 1]
        else:
            raise NotImplementedError()

    def _sub_sample(self):
        train_tokens = sum(self._counts.values())

        if self._dirty:
            sub_corp = []
            for p in self._corpus:
                sub_p = []
                for w in p:
                    if w in {self._pad_tok}:
                        sub_p.append(w)
                        continue

                    freq = self._counts[w] / train_tokens
                    if freq > self._sub:
                        prob_drop = max(0, 1 - np.sqrt(self._sub / freq))
                        if prob_drop > 0:
                            draw = np.random.rand()
                            if prob_drop > draw:
                                continue

                    sub_p.append(w)
                sub_corp.append(sub_p)

            self._corpus = [s for s in tqdm(sub_corp) if len(s) > 1]
            self._counts = dict(Counter([w for s in tqdm(self._corpus) for w in s]))
        else:
            raise NotImplementedError()

    def _truncate_vocab(self):
        if self._dirty:
            self._counts = {
                k: v for k, v in sorted(self._counts.items(), key=lambda kv: kv[1], reverse=True)[:self._max_vocab]
            }
            self._corpus = [[w for w in s if w in self._counts] for s in tqdm(self._corpus)]
            self._corpus = [s for s in tqdm(self._corpus) if len(s) > 1]
        else:
            raise NotImplementedError()

    def _generate_pairs(self):
        for tok, ctxs in tqdm(self._word_ctx.items()):
            ix = torch.LongTensor([self._token2idx[tok]])
            for ctx in ctxs:
                ictx = torch.LongTensor([self._token2idx[ic] for ic in ctx])
                weight = torch.Tensor([1 / abs(ix - self._m) for ix in range(self._m)] + [1 / (ix + 1) for ix in range(self._m)])
                co_occur = torch.Tensor([self._word_co[tok][ic] for ic in ctx])

                self._pairs.append({
                    'word': ix,
                    'context': ictx,
                    'weight': weight,
                    'x_ij': co_occur
                })

    @property
    def vocab_size(self):
        return len(self._token2idx)

    def lookup(self, tok):
        return self._token2idx[tok]

    def rlookup(self, idx):
        return self._idx2token[idx]

    def sample_tokens(self, size):
        sel = torch.LongTensor(np.random.choice(list(range(self.vocab_size)), replace=True, size=size, p=self._idx2prob))

        return sel
