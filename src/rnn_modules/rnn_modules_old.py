import os
import sys

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.insert(1, os.path.join(sys.path[0], '../'))
from src.nn_modules import data_utils

CUDA = torch.cuda.is_available()

class Config(object):
    def __init__(self, batch_size, embedding_dim, learning_rate, epochs_num):
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.epochs_num = epochs_num


class RNNClassifier(nn.Module):
    def __init__(self, embeddings, hidden_dim, config):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim

        # load embeddings
        self.embed = nn.Embedding(*embeddings.shape)
        if CUDA:
            self.embed.weight = nn.Parameter(torch.cuda.FloatTensor(embeddings))
        else:
            self.embed.weight = nn.Parameter(torch.FloatTensor(embeddings))
        self.embed.weight.requires_grad = False

        self.hidden = self.hidden_init()
        self.rnn = nn.RNN(embeddings.shape[1], hidden_dim, num_layers=1, nonlinearity='tanh',
                          dropout=0.6)
        self.linear = nn.Linear(hidden_dim, 2)

        self.softmax = nn.CrossEntropyLoss()
        self.optimizer = None

    def hidden_init(self):
        if CUDA:
            hidden = Variable(torch.zeros(1, self.config.batch_size, self.hidden_dim)).type(
                torch.cuda.FloatTensor)
        else:
            hidden = Variable(torch.zeros(1, self.config.batch_size, self.hidden_dim))
        return hidden

    def forward(self, seq_tensor, seq_lengths):
        """
        forward pass
        :param sentence (num_words, dim_embeddings) : featureized sentence
        :return: predicted logits
        """
        # embed your sequences
        seq_tensor = self.embed(seq_tensor)
        # pack them up nicely
        packed_input = pack_padded_sequence(seq_tensor, seq_lengths.cpu().numpy())
        outputs, self.hidden = self.rnn(packed_input, self.hidden)
        outputs, _ = pad_packed_sequence(outputs)
        x = self.linear(outputs[-1])
        return x

    def add_optimizer(self):
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                    lr=self.config.learning_rate)

    def loss(self, inputs, targets):
        return self.softmax(inputs, targets)


def sequence2tensor(sequence, vocab, targets=None):
    """
    :param sentence: List of sentences
    :return:
    """
    targets_parsed = []
    vectorized_seq = []
    for i, sentence in enumerate(sequence):
        word_list = []
        for word in sentence:
            try:
                word_list.append(vocab[word])
            except KeyError:
                continue
        # to avoid adding sentence with 0 tokens on the vocabulary
        if len(word_list) > 0:
            if targets is not None:
                targets_parsed.append(targets[i])
            vectorized_seq.append(word_list)
    if CUDA:
        seq_lengths = torch.cuda.LongTensor(list(map(len, vectorized_seq)))
        seq_tensor = torch.zeros((len(vectorized_seq), seq_lengths.max())).type(
            torch.cuda.LongTensor)

    else:
        seq_lengths = torch.LongTensor(list(map(len, vectorized_seq)))
        seq_tensor = torch.zeros((len(vectorized_seq), seq_lengths.max())).type(torch.LongTensor)

    for idx, (seq, seqlen) in enumerate(zip(vectorized_seq, seq_lengths)):
        try:
            if CUDA:
                seq_tensor[idx, :seqlen] = torch.cuda.LongTensor(seq)
            else:
                seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        except ValueError as e:
            # to avoid error when none of the tokens in the sentence are in the vocabulary
            continue

    return seq_tensor, seq_lengths, targets_parsed


def generate_dataloader(dataset_tensor, lengths_tensor,  labels, batch_size, ratio_train_val_set =
    0.9):
    labels_tensor = torch.FloatTensor(labels)
    if CUDA:
        labels_tensor.cuda()

    indeces = torch.randperm(len(labels))
    train_indeces = indeces[:int(len(labels)*ratio_train_val_set)]
    val_indeces = indeces[int(len(labels)*ratio_train_val_set):]

    dataset_tensor_train = dataset_tensor[train_indeces]
    dataset_tensor_val = dataset_tensor[val_indeces]

    lengths_tensor_train = lengths_tensor[train_indeces]
    lengths_tensor_val = lengths_tensor[val_indeces]

    labels_tensor_train = labels_tensor[train_indeces]
    labels_tensor_val = labels_tensor[val_indeces]



    tensor_dataset_train = torch.utils.data.TensorDataset(dataset_tensor_train,
                                                          labels_tensor_train)
    tensor_lenghts_train = torch.utils.data.TensorDataset(lengths_tensor_train,
                                                          labels_tensor_train)
    tensor_dataset_val = torch.utils.data.TensorDataset(dataset_tensor_val,
                                                          labels_tensor_val)
    tensor_lenghts_val = torch.utils.data.TensorDataset(lengths_tensor_val,
                                                          labels_tensor_val)

    dataloader_train = torch.utils.data.DataLoader(tensor_dataset_train, batch_size=batch_size,
                                                 shuffle=False)

    dataloader_val = torch.utils.data.DataLoader(tensor_dataset_val, batch_size=batch_size,
                                                 shuffle=False)

    return dataloader_train, tensor_lenghts_train, dataloader_val, tensor_lenghts_val

def sort_sequences(seq_tensor, seq_lengths):
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    seq_tensor = seq_tensor.transpose(0, 1)
    return seq_tensor, seq_lengths


def train(model, dataloader_train, train_lengths, config, dataloader_val=None,
          val_lengths=None):
    iter_counter = 0
    current_loss = 0

    train_loss_history = []
    val_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []
    model.add_optimizer()

    for epoch in range(config.epochs_num):
        for index, sample_batched in enumerate(dataloader_train):
            if len(sample_batched[1]) != dataloader_train.batch_size:
                continue
            print("iter ", index)
            seq_tensor, labels = sample_batched
            seq_lengths, _ = train_lengths[index*config.batch_size:(index+1)*config.batch_size]
            # sort tensor by length for the packing function later on
            seq_tensor, seq_lengths = sort_sequences(seq_tensor, seq_lengths)

            if CUDA:
                labels = Variable(labels.type(torch.cuda.LongTensor), requires_grad=False)
                seq_var = Variable(seq_tensor, requires_grad=False).type(torch.cuda.LongTensor)

            else:
                labels = Variable(labels.type(torch.LongTensor), requires_grad=False)
                seq_var = Variable(seq_tensor, requires_grad=False)


            model.optimizer.zero_grad()

            output = model(seq_var, seq_lengths)
            current_loss = model.loss(output, labels)

            current_loss.backward(retain_graph=True)
            model.optimizer.step()

            iter_counter += 1

        train_accuracy, train_loss = test(model, dataloader_train, train_lengths, config)
        print("epoch: {0}, train loss: {1}, train accuracy: {2}" .format(epoch, train_loss,
                                                             train_accuracy))
        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)

        if dataloader_val:
            val_accuracy, val_loss = test(model, dataloader_val, val_lengths, config)
            print("epoch: {0}, val loss: {1}, val accuracy: {2}".format(epoch, val_loss,
                                                                        val_accuracy))
            val_loss_history.append(val_loss)
            val_accuracy_history.append(val_accuracy)

    return train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history

def test(model, dataloader, lengths, config):
    loss = 0
    accuracy = 0
    num_batches = int(len(lengths)/config.batch_size)
    for index, sample_batched in enumerate(dataloader):
        if len(sample_batched[1]) != dataloader.batch_size:
            continue
        seq_tensor, labels = sample_batched
        seq_lengths, _ = lengths[index * config.batch_size:(index + 1) * config.batch_size]
        # sort tensor by length for the packing function later on
        seq_tensor, seq_lengths = sort_sequences(seq_tensor, seq_lengths)

        if CUDA:
            labels = Variable(labels.type(torch.cuda.LongTensor), requires_grad=False)
            seq_var = Variable(seq_tensor, requires_grad=False).type(torch.cuda.LongTensor)

        else:
            labels = Variable(labels.type(torch.LongTensor), requires_grad=False)
            seq_var = Variable(seq_tensor, requires_grad=False)

        output = model(seq_var, seq_lengths)
        accuracy += compute_accuracy(output, labels)
        loss += model.loss(output, labels)
    return accuracy/num_batches, loss.data.cpu().numpy()/num_batches

def compute_accuracy(output, target):
    _, predicted = torch.max(output.data, 1)
    c = (predicted == target.data).squeeze()
    return torch.sum(c)/target.size(0)

def run():
    # hyperparameters
    config = Config(batch_size=500, embedding_dim=20, learning_rate=10**-2, epochs_num=20)
    # get features and labels of tweets
    print("Loading data ... ")
    embeddings, vocabulary, dataset, labels = data_utils.load_params(config.embedding_dim,
                                                                     use_all_data=False)
    print("Vectorizing sentences ...")
    dataset_tensor, lengths_tensor, parsed_targets = sequence2tensor(dataset, vocabulary, labels)

    dataloader_train, lengths_train, dataloader_test, lengths_test \
    = generate_dataloader(dataset_tensor, lengths_tensor, parsed_targets, config.batch_size)
    # learning algorithm

    print("preparing model...")
    model = RNNClassifier(embeddings, 100, config)
    print("model loaded. Start training data...")
    train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = train(
        model, dataloader_train, lengths_train, config,
        dataloader_test, lengths_test)

if __name__ == '__main__':
    run()