import re
import torch

import numpy as np

from collections import Counter
from glob import glob
from nltk.tokenize import sent_tokenize
from tqdm import tqdm


def get_random_contexts(corpus, n, window=5, pad='longest', pytorch=True):
    centers = []
    contexts = []
    maxlen = -1

    for s in corpus.sample_sentences(n):
        # select pivot
        w_id = np.random.randint(low=0, high=len(s))
        centers.append(s[w_id])

        # construct context
        context = s[max(0, w_id - window):w_id]
        if w_id + 1 < len(s):
            context += s[w_id + 1:min(len(s), w_id + window + 1)]
        contexts.append(context)

        maxlen = max(maxlen, len(context))

    mask = []
    if pad == 'longest':
        contexts = np.array([c + (maxlen - len(c)) * [corpus._pad_tok] for c in contexts])

        centers = np.array([corpus.lookup(x) for x in centers])
        contexts = np.array([[corpus.lookup(x) for x in xs] for xs in contexts])
        mask = np.ones(contexts.shape, dtype=np.int)
        mask[contexts == corpus.lookup(corpus._pad_tok)] = 0
    else:
        centers = np.array([corpus.lookup(x) for x in centers])
        contexts = np.array([[corpus.lookup(x) for x in xs] for xs in contexts])

    if pytorch:
        centers = torch.LongTensor(centers)
        contexts = torch.LongTensor(contexts)
        mask = torch.LongTensor(mask)

    return centers, contexts, mask


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

    def __init__(self, path, min_threshold=-1, max_prob=1e-2,
                 start_token='<s>', end_token='</s>', pad_token='<pad>'):
        self._path = path
        self._min_threshold = min_threshold
        self._max_prob = max_prob

        self._start_tok = start_token
        self._end_tok = end_token
        self._pad_tok = pad_token

        self._sents = []
        self._counts = Counter()

        self._token2idx = {}
        self._idx2token = {}

        self._idx2prob = []

        self._load()

    def _load(self, lower=True):
        print('Reading raw data')
        for f in glob(self.DATA_ROOT + self._path):
            with open(f) as fp:
                for s in tqdm(sent_tokenize(fp.read())):
                    if s.strip():
                        self._sents.append(tokenize(s.strip(), lower=lower))
        print(f'Loaded {len(self._sents)} sentences')

        print('Computing token counts')
        self._counts = dict(Counter([w for s in tqdm(self._sents) for w in s]))
        print(f'Loaded {len(self._counts)} token types; {sum(self._counts.values())} token instances')

        if self._min_threshold > 0:
            print(f'Pruning rare instances (tokens occurring < {self._min_threshold} times in the corpus)')
            original_size = len(self._counts)
            self._counts = {w: cnt for w, cnt in self._counts.items() if cnt > self._min_threshold}
            new_size = len(self._counts)
            print(f'Discarded {original_size - new_size} token types!')
            print(f'Vocab is now {new_size} token types ({100 * new_size / original_size:.0f}% of original size)')

            print('Updating sentences...')
            self._sents = [[w for w in s if w in self._counts] for s in tqdm(self._sents)]

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
        self._idx2prob = [v if v < self._max_prob else 0 for v in self._idx2prob]

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

