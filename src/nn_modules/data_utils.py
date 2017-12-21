import pickle
import random

import numpy as np
import torch
import sys, os
random.seed = 10
from src.nn_modules import train_utils
# encoding=utf8
import os
import matplotlib.pyplot as plt
import csv
from torch.autograd import Variable


def load_vocabulary(directory='twitter-datasets'):
    with open(directory + '/' + 'vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    return vocab



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


def load_train_dataset(loaded_sample_pos, loaded_sample_neg):
    """
    Loads the dataset and generate corresponding labels, then it shuffles them
    :param use_all_data (bool) : indicates if all the dataset is used
    :return: shuffled_dataset (list) : dataset containing the sentences
            shuffled_lables (np.array): labels of the dataset,
                                        0 relate to negative and 1 to positive
    """


    dataset_pos = parse_samples(loaded_sample_pos)
    labels_pos = np.ones(len(dataset_pos))
    print(np.shape(labels_pos))
    dataset_neg = parse_samples(loaded_sample_neg)
    labels_neg = np.zeros(len(dataset_neg))
    print(np.shape(labels_neg))
    # join
    dataset = dataset_pos + dataset_neg
    labels = np.vstack((labels_pos, labels_neg)).flatten()

    return dataset, labels

def load_samples(filename, directory='twitter-datasets-pp'):
    with open(directory + '/' + filename, encoding="utf-8") as file:
        dataset = file.readlines()
    return dataset

def load_test_data():
    filaname = 'test_data_NOnum.txt'
    dataset_test = parse_samples(load_samples(filaname))
    return dataset_test


def load_params(use_all_data=False, embeddings_dir='embeddings', embeddings_name='embeddings', vocab_directory='twitter-datasets'):
    """
    Load parameter necessary for training.
    :param use_all_data (bool)
    :param directory: directory of data
    :return: embeddings, vocabulary, dataset, labels
    """
    embeddings = np.load(embeddings_dir + '/' + embeddings_name)
    with open(vocab_directory + '/' + 'vocab.pkl', 'rb') as file:
        vocabulary = pickle.load(file)
    dataset, labels = load_train_dataset(use_all_data)
    return embeddings, vocabulary, dataset, labels

def save_embedded_dataset(dataset, labels, filename='00',):
    directory = 'saved_models'
    with open(directory + '/' + 'dataset_' + filename + '.pkl', 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

    with open(directory + '/' + 'labels_' + filename + '.pkl', 'wb') as f:
        pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)

def load_embedded_dataset(filename='00'):
    directory = 'saved_models'
    with open(directory + '/' + 'dataset_' + filename + '.pkl', 'rb') as f:
        dataset = pickle.load(f)

    with open(directory + '/' + 'labels_' + filename + '.pkl', 'rb') as f:
        labels = pickle.load(f)

    return dataset, labels


def generate_dataloader(dataset_features, labels, batch_size, ratio_train_val_set = 0.9):
    tensor_features = torch.FloatTensor(dataset_features)
    labels_tensor = torch.FloatTensor(labels)

    indeces = torch.randperm(len(labels))
    train_indeces = indeces[:int(len(labels)*ratio_train_val_set)]
    val_indeces = indeces[int(len(labels)*ratio_train_val_set):]

    tensor_features_train = tensor_features[train_indeces]
    tensor_features_val = tensor_features[val_indeces]
    labels_tensor_train = labels_tensor[train_indeces]
    labels_tensor_val = labels_tensor[val_indeces]

    tensor_dataset_train = torch.utils.data.TensorDataset(tensor_features_train,
                                                          labels_tensor_train)
    tensor_dataset_val = torch.utils.data.TensorDataset(tensor_features_val,
                                                          labels_tensor_val)
    dataloader_train = torch.utils.data.DataLoader(tensor_dataset_train, batch_size=batch_size,
                                                 shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(tensor_dataset_val, batch_size=batch_size,
                                                 shuffle=True)
    return dataloader_train, dataloader_test

def plot_convergence(epochs_num, train_losses, validation_losses, directory):
    """Plot test and train error against the epoch"""
    fig, ax = plt.subplots()
    x = np.arange(0, epochs_num)
    train_trend, = ax.plot(x, train_losses, label="Train loss")
    test_trend, = ax.plot(x, validation_losses, label="Test loss")
    ax.legend(loc='lower right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss history')
    label = "convergence plot"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + '/' + label + '.png',
                bbox_inches='tight')

def plot_accuracy(epochs_num, train_accuracies, validation_accuracies, directory):
    fig, ax = plt.subplots()
    x = np.arange(0, epochs_num)
    train_trend, = ax.plot(x, train_accuracies, label="Train accuracy")
    test_trend, = ax.plot(x, validation_accuracies, label="Test accuracy")
    ax.legend(loc='lower right')
    plt.xlabel('accuracy')
    plt.ylabel('loss')
    plt.title('Learning curves')
    label = "accuracy plot"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + '/' + label + '.png',
                bbox_inches='tight')


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


def format(value):
    return "%.3f" % value


def create_csv_submission(model, dataset_features, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    tensor_features = torch.FloatTensor(dataset_features)
    dataset = Variable(tensor_features, requires_grad=False)
    output = model(dataset)
    y_pred = train_utils.predict(output.data).numpy()
    y_pred = np.where(y_pred == 0, -1, 1)
    with open( name + "/" + "submission_file.csv", 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for idx, pred in enumerate(y_pred):
            writer.writerow({'Id': idx + 1, 'Prediction': int(pred)})



def average_sentece_length(dataset):
    lengths = [len(sentence) for sentence in dataset]
    average = sum(lengths)/len(lengths)
    return average, max(lengths)


