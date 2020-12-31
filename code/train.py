import json
import os
import pickle
import torch
import random

import numpy as np
import wandb as wb

from argparse import ArgumentParser

from modeling.glove import GloVe
from modeling.word2vec import CBOW, Skipgram, SkipgramNS, CBOWNS

from text_utils import Corpus


def train_w2v():
    parser = ArgumentParser()

    parser.add_argument('--input', default='*.txt', type=str,
                        help='Input search pattern to construct training data from')
    parser.add_argument('--output', default='models/', type=str,
                        help='Where to save models to. (Default: "models/"')
    parser.add_argument('--save_every', default=1, type=int,
                        help='Epochs between weight saves (Default: 1).')

    parser.add_argument('--min_freq', default=-1, type=int,
                        help='Minimum word occurrences to retain. (Default: 0, no pruning of rare symbols)')
    parser.add_argument('--max_vocab', default=-1, type=int,
                        help='Minimum vocabulary size. (Default: -1, no limit)')
    parser.add_argument("--no-lower", default=False, action="store_true",
                        help="Disables lower-casing that occurs when constructing the vocabulary")
    parser.add_argument('--sample_thresh', default=0, type=float,
                        help='A sub-sampling threshold used in word2vec to drop frequent words in some instances. (Default: 0)')
    parser.add_argument('--context', default=10, type=int,
                        help='How large of a context window to use for word2vec? (Default: 10)')

    parser.add_argument('--w2v', default='SG', type=str,
                        help='Which word2vec sub-algorithm to select? [CBOW,SG]. (Default: SG)')
    parser.add_argument('--ns', default=0, type=int,
                        help='Number of negative samples to use for word2vec. (Default: 0 or naive softmax)')
    parser.add_argument('--hidden_dim', default=16, type=int,
                        help='The size of the learned representation. (Default: 128)')
    parser.add_argument('--batch', default=64, type=int,
                        help='Batch size for training. (Default: 64)')
    parser.add_argument('--epochs', default=3, type=int,
                        help='Number of training epochs. (Default: 3)')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='Learning rate for Adagrad optimizer. (Default: 0.01)')

    parser.add_argument('--seed', default=0, help='Random seed. (Default: 0)')

    args = parser.parse_args()

    model_name = f'word2vec_{args.w2v}{"NS_" + str(args.ns) if args.ns > 0 else ""}_{args.context}_{args.hidden_dim}'
    outpath = f'{args.output}{model_name}/'
    os.makedirs(outpath, exist_ok=True)
    json.dump(vars(args), open(f'{outpath}args.json', 'w+'))

    wb.init(project='wordvectors', name=model_name, config=vars(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    corpus = Corpus(args.input,
                    min_freq=args.min_freq,  # minimum number of times for a word type to be retained
                    max_vocab=args.max_vocab,  # maximum size of vocabulary
                    sub=args.sample_thresh,  # sub-sampling threshold
                    m=args.context,  # context window size
                    lower=(not args.no_lower),
                    dirty=True,  # if true, will drop words prior to slicing context windows (otherwise after)
                    dyn=False,  # if true, window is maximal (and randomly sampled each generation)
                    )

    pickle.dump(corpus, open(f'{outpath}corpus.pkl', 'wb+'))

    hidden_dim = args.hidden_dim
    batch_size = args.batch
    max_epochs = args.epochs

    if args.w2v == 'SG':
        model = Skipgram(corpus.vocab_size, hidden_dim) if args.ns == 0 else SkipgramNS(corpus.vocab_size,
                                                                                        hidden_dim)
    elif args.w2v == 'CBOW':
        model = CBOW(corpus.vocab_size, hidden_dim) if args.ns == 0 else CBOWNS(corpus.vocab_size, hidden_dim)
    else:
        raise ValueError(f'Unrecognized word2vec selection: {args.w2v}')

    opt = torch.optim.Adagrad(params=list(model.parameters()), lr=args.lr)

    if args.ns == 0:
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
        model.train_embedding(corpus, opt, loss_func, iterations=max_epochs, batch_size=batch_size, outpath=outpath,
                              save_every=args.save_every)
    else:
        loss_func = torch.nn.BCEWithLogitsLoss(reduction='sum')
        model.train_embedding(corpus, opt, loss_func,
                              iterations=max_epochs, batch_size=batch_size, neg_samples=args.ns, outpath=outpath,
                              save_every=args.save_every)


def train_glove():
    parser = ArgumentParser()

    parser.add_argument('--input', default='*.txt', type=str,
                        help='Input search pattern to construct training data from')
    parser.add_argument('--output', default='models/', type=str,
                        help='Where to save models to. (Default: "models/"')
    parser.add_argument('--save_every', default=1, type=int,
                        help='Epochs between weight saves (Default: 1).')

    parser.add_argument('--min_freq', default=-1, type=int,
                        help='Minimum word occurrences to retain. (Default: 0, no pruning of rare symbols)')
    parser.add_argument('--max_vocab', default=-1, type=int,
                        help='Minimum vocabulary size. (Default: -1, no limit)')
    parser.add_argument("--no-lower", default=False, action="store_true",
                        help="Disables lower-casing that occurs when constructing the vocabulary")
    parser.add_argument('--sample_thresh', default=0, type=float,
                        help='A sub-sampling threshold used in word2vec to drop frequent words in some instances. (Default: 0)')
    parser.add_argument('--context', default=10, type=int,
                        help='How large of a context window to use for word2vec? (Default: 10)')

    parser.add_argument('--hidden_dim', default=16, type=int,
                        help='The size of the learned representation. (Default: 128)')

    parser.add_argument('--batch', default=64, type=int,
                        help='Batch size for training. (Default: 64)')
    parser.add_argument('--epochs', default=3, type=int,
                        help='Number of training epochs. (Default: 3)')
    parser.add_argument('--lr', default=0.05, type=float,
                        help='Learning rate for Adagrad optimizer. (Default: 0.05)')

    parser.add_argument('--seed', default=0, help='Random seed. (Default: 0)')

    args = parser.parse_args()

    model_name = f'glove_{args.context}_{args.hidden_dim}'
    outpath = f'{args.output}{model_name}/'
    os.makedirs(outpath, exist_ok=True)
    json.dump(vars(args), open(f'{outpath}args.json', 'w+'))

    wb.init(project='wordvectors', name=model_name, config=vars(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    corpus = Corpus(args.input,
                    min_freq=args.min_freq,  # minimum number of times for a word type to be retained
                    max_vocab=args.max_vocab,  # maximum size of vocabulary
                    sub=args.sample_thresh,  # sub-sampling threshold
                    m=args.context,  # context window size
                    lower=(not args.no_lower),
                    dirty=True,  # if true, will drop words prior to slicing context windows (otherwise after)
                    dyn=False,  # if true, window is maximal (and randomly sampled each generation)
                    )

    pickle.dump(corpus, open(f'{outpath}corpus.pkl', 'wb+'))

    model = GloVe(corpus.vocab_size, args.hidden_dim)
    opt = torch.optim.Adagrad(params=list(model.parameters()), lr=args.lr)

    model.train_embedding(corpus, opt, iterations=args.epochs, batch_size=args.batch, outpath=outpath,
                          save_every=args.save_every)


if __name__ == '__main__':
    # train_w2v()
    train_glove()

