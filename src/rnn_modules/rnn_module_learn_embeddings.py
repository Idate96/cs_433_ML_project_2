import csv

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.nn_modules import data_utils


class Config(object):
    def __init__(self, batch_size, embedding_dim, vocab_size, learning_rate, epochs_num):
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.epochs_num = epochs_num


class RNNClassifier(nn.Module):
    def __init__(self, hidden_dim, config):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(config.vocab_size, embedding_dim=config.embedding_dim)
        self.rnn = nn.RNN(config.embedding_dim, hidden_dim, num_layers=1, nonlinearity='tanh',
                          dropout=0.2)
        self.linear = nn.Linear(hidden_dim, 2)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, seq_tensor, seq_lengths):
        embedding = self.embedding(seq_tensor)
        padded_embedding = pack_padded_sequence(embedding, seq_lengths.numpy(), batch_first=True)
        outputs, _ = self.rnn(padded_embedding)
        outputs, lengths = pad_packed_sequence(outputs, batch_first=False)
        lengths = [l - 1 for l in lengths]
        last_output = outputs[lengths, range(len(lengths))]
        logits = self.linear(last_output)
        return logits

def sort_sequences(seq_tensor, labels, seq_lengths):
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    labels = labels[perm_idx]
    return seq_tensor, labels, seq_lengths

def generate_dataloader(dataset_tensor, lengths_tensor,  labels, batch_size, ratio_train_val_set =
    0.9):
    labels_tensor = torch.FloatTensor(labels)
    indeces = torch.randperm(len(labels))

    train_indeces = indeces[:int(len(labels)*ratio_train_val_set)]
    val_indeces = indeces[int(len(labels)*ratio_train_val_set):]

    dataset_tensor_train = dataset_tensor[train_indeces]
    lengths_tensor_train = lengths_tensor[train_indeces]
    dataset_tensor_val = dataset_tensor[val_indeces]
    labels_tensor_train = labels_tensor[train_indeces]
    labels_tensor_val = labels_tensor[val_indeces]
    lengths_tensor_val = lengths_tensor[val_indeces]


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
        if word_list:
            if targets is not None:
                targets_parsed.append(targets[i])
            vectorized_seq.append(word_list)

    seq_lengths = torch.LongTensor(list(map(len, vectorized_seq)))
    seq_tensor = torch.zeros((len(vectorized_seq), seq_lengths.max())).type(torch.LongTensor)

    for idx, (seq, seqlen) in enumerate(zip(vectorized_seq, seq_lengths)):
        try:
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        except ValueError as e:
            # to avoid error when none of the tokens in the sentence are in the vocabulary
            continue
    return seq_tensor, seq_lengths, targets_parsed


def train(model, dataloader_train, train_lengths, config, dataloader_val=None,
          val_lengths=None):

    optimizer = optim.Adam(model.parameters())
    loss = nn.CrossEntropyLoss()

    iter_num = 0
    for epoch in range(config.epochs_num):
        print("epoch : ", epoch)
        model.train()
        for index, batch in enumerate(dataloader_train):
            if len(batch[1]) != dataloader_train.batch_size:
                continue
            sequence, labels = batch
            sequence_lengths, _ = train_lengths[index*config.batch_size:(index+1)*config.batch_size]
            sequence, labels, sequence_lengths = sort_sequences(sequence, labels, sequence_lengths)

            labels_var = Variable(labels.type(torch.LongTensor), requires_grad=False)
            sequence_var = Variable(sequence.type(torch.LongTensor), requires_grad=False)
            # sequence_lengths_var = Variable(sequence_lengths.type(torch.LongTensor),
            #                                 requires_grad=False)

            optimizer.zero_grad()
            logits = model(sequence_var, sequence_lengths)
            loss_value = loss(logits, labels_var)

            loss_value.backward()
            optimizer.step()

            accuracy = compute_accuracy(logits, labels_var)

            iter_num += 1
            if iter_num % 50 == 0:
                print("iter {0}, train loss: {1}, accuracy {2}" .format(
                    iter_num, loss_value.data.numpy(), accuracy))

        model.eval()
        train_loss, train_accuracy = test(model, dataloader_train, train_lengths, config)
        print("epoch: {0}, train loss: {1}, train accuracy: {2}".format(epoch, train_loss,
                                                                        train_accuracy.data.numpy()))

        if dataloader_val:
            val_loss, val_accuracy = test(model, dataloader_val, val_lengths, config)
            print("epoch: {0}, val loss: {1}, val accuracy: {2}".format(epoch, val_loss,
                                                                        val_accuracy.data.numpy()))

def test(model, dataloader, lengths, config):
    loss_value = 0
    loss = nn.CrossEntropyLoss()

    accuracy = 0
    num_batches = 0
    for index, batch in enumerate(dataloader):
        if len(batch[1]) != dataloader.batch_size:
            continue
        sequence, labels = batch
        sequence_lengths, _ = lengths[
                              index * config.batch_size:(index + 1) * config.batch_size]
        sequence, labels, sequence_lengths = sort_sequences(sequence, labels, sequence_lengths)

        labels_var = Variable(labels.type(torch.LongTensor), requires_grad=False)
        sequence_var = Variable(sequence.type(torch.LongTensor), requires_grad=False)

        logits = model(sequence_var, sequence_lengths)

        loss_value += loss(logits, labels_var)
        accuracy += compute_accuracy(logits, labels_var)
        num_batches += 1

    return loss_value/num_batches, accuracy/num_batches


def compute_accuracy(output, target):
    _, predicted = torch.max(output.data, 1)
    c = (predicted == target.data).squeeze()
    return torch.sum(c)/target.size(0)

def create_submission_rnn(model, seq_tensor, seq_lenghts, name):

    y_pred = np.where(y_pred == 0, -1, 1)
    with open( name + "/" + "submission_file", 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for idx, pred in enumerate(y_pred):
            writer.writerow({'Id': idx + 1, 'Prediction': int(pred)})


def run():
    # hyperparameters
    config = Config(batch_size=500, embedding_dim=200, vocab_size=21161, learning_rate=10**-3, \
                                                                               epochs_num=20)
    # get features and labels of tweets
    print("Loading data ... ")
    embeddings, vocabulary, dataset, labels = data_utils.load_params(config.embedding_dim,
                                                                     use_all_data=False)
    print("Vectorizing sentences ...")
    dataset_tensor, lengths_tensor, parsed_targets = sequence2tensor(dataset, vocabulary, labels)

    dataloader_train, lengths_train, dataloader_test, lengths_test = generate_dataloader(
        dataset_tensor, lengths_tensor, parsed_targets, config.batch_size)

    model = RNNClassifier(hidden_dim=100, config=config)

    train(model, dataloader_train, lengths_train, config, dataloader_test, lengths_test)

if __name__ == '__main__':
    run()



