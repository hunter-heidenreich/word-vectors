import pickle
import torch
import random

import numpy as np
import wandb as wb

from argparse import ArgumentParser

from modeling.word2vec import CBOW, Skipgram, SkipgramNS, CBOWNS
from text_utils import Corpus


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--min_thresh', default=0, type=int,
                        help='Minimum threshold for word occurrences. (Default: 0, no pruning of rare symbols)')
    parser.add_argument('--max_thresh', default=-1, type=int,
                        help='Minimum vocabulary size. (Default: -1, no limit)')
    parser.add_argument('--input', default='*.txt', type=str,
                        help='Input search pattern to construct training data from')
    parser.add_argument('--output', default='models/', type=str,
                        help='Where to save models to. (Default: "models/"')
    parser.add_argument("--no-lower", default=False, action="store_true",
                        help="Disables lower-casing that occurs when constructing the vocabulary")
    parser.add_argument('--sample_thresh', default=0, type=float,
                        help='A sub-sampling threshold used in word2vec to drop frequent words in some instances. (Default: 0)')
    parser.add_argument('--alg', default='w2v', type=str,
                        help='Which algorithm to train? [w2v]. (Default: w2v)')
    parser.add_argument('--w2v', default='SG', type=str,
                        help='Which word2vec sub-algorithm to select? [CBOW,SG]. (Default: SG)')
    parser.add_argument('--ns', default=0, type=int,
                        help='Number of negative samples to use for word2vec. (Default: 0 or naive softmax)')
    parser.add_argument('--context', default=10, type=int,
                        help='How large of a context window to use for word2vec? (Default: 10)')
    parser.add_argument('--hidden_dim', default=128, type=int,
                        help='The size of the learned representation. (Default: 128)')
    parser.add_argument('--batch', default=64, type=int,
                        help='Batch size for training. (Default: 64)')
    parser.add_argument('--steps', default=10_000, type=int,
                        help='Number of training batches/steps. (Default: 10,000)')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate for AdamW optimizer. (Default: 1e-3)')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='Beta1 for averaging of gradient in Adam. (Default: 0.9)')
    parser.add_argument('--beta2', default=0.999, type=float,
                        help='Beta2 for averaging of squared gradient in Adam. (Default: 0.999)')
    parser.add_argument('--wd', default=1e-2, type=float,
                        help='Weight decay for Adam optimizer. (Default: 1e-2)')
    parser.add_argument('--seed', default=0, help='Random seed. (Default: 0)')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    corpus = Corpus(args.input,
                    min_threshold=args.min_thresh,
                    max_threshold=args.max_thresh,
                    lower=(not args.no_lower),
                    sample_thresh=args.sample_thresh)

    hidden_dim = args.hidden_dim
    batch_size = args.batch
    context_size = args.context
    max_steps = args.steps

    if args.alg == 'w2v':
        model_name = f'word2vec_{args.w2v}{"NS_" + str(args.ns) if args.ns > 0 else ""}_{args.context}_{args.hidden_dim}'
        wb.init(project='wordvectors', name=model_name, config=vars(args))
        if args.ns == 0:
            if args.w2v == 'SG':
                model = Skipgram(corpus.vocab_size, hidden_dim)
            elif args.w2v == 'CBOW':
                model = CBOW(corpus.vocab_size, hidden_dim)
            else:
                raise ValueError(f'Unrecognized word2vec selection: {args.w2v}')
            loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
        else:
            if args.w2v == 'SG':
                model = SkipgramNS(corpus.vocab_size, hidden_dim)
            elif args.w2v == 'CBOW':
                model = CBOWNS(corpus.vocab_size, hidden_dim)
            else:
                raise ValueError(f'Unrecognized word2vec selection: {args.w2v}')
            loss_func = torch.nn.BCEWithLogitsLoss(reduction='sum')

        opt = torch.optim.AdamW(
            params=list(model.parameters()), lr=args.lr, betas=(args.beta1, args.beta2),
            weight_decay=args.wd
        )

        model.train_embedding(corpus, opt, loss_func,
                              iterations=max_steps,
                              batch_size=batch_size,
                              window_len=context_size)

        # Saving model and corpus
        torch.save(model, f'{args.output}{model_name}_{max_steps}_sub.pt')
        pickle.dump(corpus, open(args.output + model_name + '_sub.vocab', 'wb'))
    else:
        raise ValueError(f'Unrecognized algorithm selection: {args.alg}')
