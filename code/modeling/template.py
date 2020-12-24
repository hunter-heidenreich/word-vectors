import torch


class ContextIndependentWordVector(torch.nn.Module):

    """
    A class that all context-independent word vectors
    may inherit from
    """

    def __init__(self, name):
        super(ContextIndependentWordVector, self).__init__()

        self._name = name

    def forward(self, input):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    def get_embedding(self):
        raise NotImplementedError

    def train_embedding(self, **args):
        raise NotImplementedError
