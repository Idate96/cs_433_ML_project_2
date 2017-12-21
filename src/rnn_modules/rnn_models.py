"""This file contains the RNN model used for the task"""

import numpy as np
import torch
from torch import nn
np.random.seed(5)
torch.manual_seed(40)
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable


class RNNClassifier(nn.Module):
    def __init__(self, hidden_dim, config, embeddings, label=''):
        super().__init__()
        self.label = label
        self.config = config
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.rnn_type = 'RNN'
        self.embedding = nn.Embedding(config.vocab_size, embedding_dim=config.embedding_dim)
        # use GloVe embeddings
        self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.embedding.weight.requires_grad = True

        self.rnn = nn.RNN(config.embedding_dim, hidden_dim, num_layers=self.num_layers,
                          nonlinearity='tanh',
                          dropout=0.5, bidirectional=False)
        self.linear = nn.Linear(hidden_dim, 2)

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, seq_tensor, seq_lengths):
        embedding = self.embedding(seq_tensor)
        padded_embedding = pack_padded_sequence(embedding, seq_lengths.numpy(), batch_first=True)
        outputs, hidden = self.rnn(padded_embedding)
        hidden.detach()
        outputs, lengths = pad_packed_sequence(outputs, batch_first=False)
        lengths = [l - 1 for l in lengths]
        last_output = outputs[lengths, range(len(lengths))]
        logits = self.linear(last_output)
        return logits

    def init_hidden(self):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.num_layers, self.config.batch_size, self.hidden_dim).zero_()),
                    Variable(weight.new(self.num_layers, self.config.batch_size, self.hidden_dim).zero_()))
        else:
            return Variable(weight.new(self.num_layers, self.config.batch_size, self.hidden_dim).zero_())

class LSTMClassifier(nn.Module):
    def __init__(self, hidden_dim, config, embeddings, label=''):
        super().__init__()
        self.label = label
        self.config = config
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.rnn_type = 'LSTM'
        self.embedding = nn.Embedding(config.vocab_size, embedding_dim=config.embedding_dim)
        # pretrained_weight is a numpy matrix of shape (num_embeddings, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.embedding.weight.requires_grad = True
        self.lstm = nn.LSTM(config.embedding_dim, hidden_dim, num_layers=self.num_layers,

                          dropout=0.5, bidirectional=False)
        self.linear = nn.Linear(hidden_dim, 2)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, seq_tensor, seq_lengths):
        embedding = self.embedding(seq_tensor)
        padded_embedding = pack_padded_sequence(embedding, seq_lengths.numpy(), batch_first=True)
        outputs, hidden_c = self.lstm(padded_embedding)
        hidden, c = hidden_c
        hidden.detach()
        c.detach()
        outputs, lengths = pad_packed_sequence(outputs, batch_first=False)
        lengths = [l - 1 for l in lengths]
        last_output = outputs[lengths, range(len(lengths))]
        logits = self.linear(last_output)
        return logits



