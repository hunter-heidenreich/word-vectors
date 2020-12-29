import re
import torch

import numpy as np

from collections import Counter
from glob import glob
from nltk.tokenize import sent_tokenize
from tqdm import tqdm


def tokenize(text, space=False, lower=False):
    text = re.sub("[ ]+", " ", text)
    tokens = []
    for token in re.split("([0-9a-zA-Z\'\-]+)", text):
        if re.search("[0-9a-zA-Z\'\-]", token):
            if lower:
                tokens.append(token.lower())
            else:
                tokens.append(token)
        elif space:
            tokens.extend(token)
    return tokens


class Corpus:

    DATA_ROOT = 'data/'

    def __init__(self, path, min_threshold=-1, max_threshold=-1, lower=True, sample_thresh=1e-5,
                 start_token='<s>', end_token='</s>', pad_token='<pad>'):
        self._path = path
        self._min_threshold = min_threshold
        self._max_threshold = max_threshold
        self._sample_thresh = sample_thresh
        self._lower = lower

        self._start_tok = start_token
        self._end_tok = end_token
        self._pad_tok = pad_token

        self._sents = []
        self._counts = Counter()

        self._token2idx = {}
        self._idx2token = {}

        self._idx2prob = []

        self._load()

    def _load(self):
        print('Reading raw data')
        for f in glob(self.DATA_ROOT + self._path):
            with open(f) as fp:
                for s in tqdm(sent_tokenize(fp.read())):
                    if s.strip():
                        self._sents.append(tokenize(s.strip(), lower=self._lower))
        print(f'Loaded {len(self._sents)} sentences')

        print('Computing token counts')
        self._counts = dict(Counter([w for s in tqdm(self._sents) for w in s]))
        train_tokens = sum(self._counts.values())
        print(f'Loaded {len(self._counts)} token types; {train_tokens} token instances')
        original_size = len(self._counts)
        original_train_tokens = train_tokens

        # min threshold pruning
        if self._min_threshold > 0:
            print(f'Pruning rare instances (tokens occurring <= {self._min_threshold} times in the corpus)')

            self._counts = {w: cnt for w, cnt in self._counts.items() if cnt > self._min_threshold}
            new_size = len(self._counts)
            train_tokens = sum(self._counts.values())

            print(f'Discarded {original_size - new_size} token types!')
            print(f'Vocab is now {new_size} token types ({100 * new_size / original_size:.0f}% of original types)')
            print(f'Corpus is now {train_tokens} tokens ({100 * train_tokens / original_train_tokens:.0f}% of original corpus)')

            print('Updating sentences...')
            self._sents = [[w for w in s if w in self._counts] for s in tqdm(self._sents)]

        if self._sample_thresh > 0:
            print(f'Sub-sampling words at a occurrence threshold of {self._sample_thresh}')
            pruned_sents = []
            for sent in self._sents:
                adj_sent = []
                for w in sent:
                    prob_drop = max(0, 1 - np.sqrt(self._sample_thresh / (self._counts[w] / train_tokens)))

                    if prob_drop > 0:
                        draw = np.random.rand()
                        if prob_drop > draw:
                            continue

                    adj_sent.append(w)

                pruned_sents.append(adj_sent)

            self._sents = pruned_sents
            self._counts = dict(Counter([w for s in tqdm(self._sents) for w in s]))
            train_tokens = sum(self._counts.values())
            new_size = len(self._counts)

            print(f'Vocab is now {new_size} token types ({100 * new_size / original_size:.0f}% of original types)')
            print(f'Corpus is now {train_tokens} tokens ({100 * train_tokens / original_train_tokens:.0f}% of original corpus)')

        if 0 < self._max_threshold < len(self._counts):
            print(f'Reducing vocabulary to top {self._max_threshold} terms')
            self._counts = {
                k: v for k, v in sorted(self._counts.items(), key=lambda kv: kv[1], reverse=True)[:self._max_threshold]
            }

            train_tokens = sum(self._counts.values())
            new_size = len(self._counts)

            print(f'Vocab is now {new_size} token types ({100 * new_size / original_size:.0f}% of original types)')
            print(f'Corpus is now {train_tokens} tokens ({100 * train_tokens / original_train_tokens:.0f}% of original corpus)')

            print('Updating sentences...')
            self._sents = [[w for w in s if w in self._counts] for s in tqdm(self._sents)]
            self._sents = [s for s in self._sents if s]

        print('Tagging sentences with pre- and post- sequence tokens')
        self._sents = np.array([[self._start_tok] + s + [self._end_tok] for s in tqdm(self._sents)], dtype=object)

        print('Computing token lookup')
        self._token2idx = {
            t: idx for idx, (t, _) in enumerate(sorted(self._counts.items(), reverse=True, key=lambda kv: kv[1]))
        }
        self._token2idx[self._start_tok] = len(self._token2idx)
        self._token2idx[self._end_tok] = len(self._token2idx)
        self._token2idx[self._pad_tok] = len(self._token2idx)

        self._idx2token = {v: k for k, v in self._token2idx.items()}

        base = 0
        for i in range(self.vocab_size):
            tok = self._idx2token[i]

            if tok in self._counts:
                val = self._counts[tok] ** 0.75
                base += val
                self._idx2prob.append(val)
            else:
                self._idx2prob.append(0)

        self._idx2prob = [(v/base) for v in self._idx2prob]
        # self._idx2prob = [v if v < self._max_prob else 0 for v in self._idx2prob]

    def sample_sentences(self, n):
        return np.random.choice(self._sents, size=n, replace=True)

    @property
    def vocab_size(self):
        return len(self._token2idx)

    def lookup(self, tok):
        return self._token2idx[tok]

    def rlookup(self, idx):
        return self._idx2token[idx]

    def sample_tokens(self, size):
        return np.random.choice(list(range(self.vocab_size)), replace=True, size=size)

