import torch

from argparse import ArgumentParser

from modeling.word2vec import Skipgram
from text_utils import Corpus, get_random_contexts


if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument('--')

    min_thresh = 1  # drop words that only occur once
    corpus = Corpus('84.txt', min_threshold=min_thresh)
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)

    vocab_size = corpus.vocab_size
    hidden_dim = 256
    batch_size = 256
    context_size = 10
    max_steps = 1_000
    learning_rate = 0.003

    model = Skipgram(vocab_size, hidden_dim)
    opt = torch.optim.SGD(params=list(model.parameters()), lr=learning_rate)

    model = torch.load('model.pyx')

    model.train_embedding(corpus, get_random_contexts, opt, loss_func,
                          iterations=max_steps, batch_size=batch_size, window_len=context_size)

    import pdb
    pdb.set_trace()


