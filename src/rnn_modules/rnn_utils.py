import csv
import os
import numpy as np
import torch
from torch.autograd import Variable


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
        else:
            vectorized_seq.append([1])

    seq_lengths = torch.LongTensor(list(map(len, vectorized_seq)))
    seq_tensor = torch.zeros((len(vectorized_seq), seq_lengths.max())).type(torch.LongTensor)

    for idx, (seq, seqlen) in enumerate(zip(vectorized_seq, seq_lengths)):
        try:
            seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        except ValueError as e:
            # to avoid error when none of the tokens in the sentence are in the vocabulary
            continue
    return seq_tensor, seq_lengths, targets_parsed


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


def sort_sequences(seq_tensor, seq_lengths, labels=None):
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    if labels is not None:
        labels = labels[perm_idx]
    return seq_tensor, seq_lengths, labels


class Config(object):
    def __init__(self, batch_size, embedding_dim, vocab_size, learning_rate, epochs_num,
                 resume=False, directory='results'):
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        self.epochs_num = epochs_num
        self.resume = resume
        self.directory = directory


def create_submission_rnn(model, seq_tensor, seq_lengths, name):
    sequence, seq_lengths, _ = sort_sequences(seq_tensor, seq_lengths)
    sequence_var = Variable(seq_tensor.type(torch.LongTensor), requires_grad=False)
    output = model(sequence_var, seq_lengths)
    _, y_pred = torch.max(output.data, 1)
    y_pred = np.where(y_pred.numpy() == 0, -1, 1)
    with open(name + "/" + "submission_file.csv", 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for idx, pred in enumerate(y_pred):
            writer.writerow({'Id': idx + 1, 'Prediction': int(pred)})


def compute_accuracy(output, target):
    _, predicted = torch.max(output.data, 1)
    c = (predicted == target.data).squeeze()
    return torch.sum(c)/target.size(0)


def save_data(epochs_num, train_accuracies, val_accuracies, train_losses, val_losses, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file = open(directory + "/data.txt", "w")
    file.write("Number of epochs: " + str(epochs_num) + "\n")
    file.write("Train set losses: \n")
    file.write(str([format(loss) for loss in train_losses]) + "\n")
    file.write("Train set accuracies: \n")
    file.write(str([format(accuracy) for accuracy in train_accuracies]) + "\n")
    file.write("Validation set losses: \n")
    file.write(str([format(loss) for loss in val_losses]) + "\n")
    file.write("Validation set accuracies: \n")
    file.write(str([format(accuracy) for accuracy in val_accuracies]) + "\n")
    file.close()

def parse_samples(dataset):
    """
    Split sentences into list of words
    :param dataset: list of sentences
    :return: parsed_dataset: a list of list of words
    """
    parsed_dataset = []
    for sentence in dataset:
        parsed_sentence = sentence.replace("\n", "").split(" ")
        parsed_dataset.append(parsed_sentence)
    return parsed_dataset


def load_samples(filename, directory='twitter-datasets'):
    with open(directory + '/' + filename, encoding="utf-8") as file:
        dataset = file.readlines()
    return dataset


def load_test_data():
    filaname = 'test_data.txt'
    dataset_test = parse_samples(load_samples(filaname))
    return dataset_test


def save_checkpoint(model, optimizer, epoch, filename):
    state_dict = {'epoch': epoch + 1,
                  'state_dict': model.state_dict(),
                  'optimizer' : optimizer.state_dict()}
    torch.save(state_dict, filename)


def resume(model, filename):
    try:
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    except FileNotFoundError as e:
        print("Checkpoint not found")
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    model.optimizer.load_state_dict(checkpoint['optimizer'])
    return model, epoch