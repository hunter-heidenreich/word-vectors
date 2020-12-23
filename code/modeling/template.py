import torch


class ContextIndependentWordVector(torch.nn.Module):

    """
    A class that all context-independent word vectors
    may inherit from
    """

    def __init__(self):
        super(ContextIndependentWordVector, self).__init__()

    def get_embedding(self):
        raise NotImplementedError

    def train_embedding(self):
        raise NotImplementedError
