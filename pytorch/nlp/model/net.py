"""Embedding + LSTM token tagger. Padding labels are -1."""
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim,
                                      padding_idx=params.pad_index)
        self.lstm = nn.LSTM(params.embedding_dim, params.lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(params.lstm_hidden_dim, params.number_of_tags)

    def forward(self, tokens):
        # Unidirectional LSTM; right padding cannot affect earlier valid tokens.
        states, _ = self.lstm(self.embedding(tokens))
        return self.fc(states).reshape(-1, self.fc.out_features)


def loss_fn(outputs, labels):
    return F.cross_entropy(outputs, labels.reshape(-1), ignore_index=-1)


def accuracy(outputs, labels):
    labels = labels.reshape(-1)
    mask = labels >= 0
    return float(np.mean(np.argmax(outputs, axis=1)[mask] == labels[mask]))


metrics = {'accuracy': accuracy}
