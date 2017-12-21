import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable
from src.nn_modules import data_utils
np.seterr(all='warn', over='raise')

def embed_sentence(sentence, vocabulary, embedding):
    """
    Embeds the sentence using GLOVE
    :param sentence: a list of words (strings)
    :param vocabulary: hash table (dictionary) maps word to an index of word in embedding matrix
    :param embedding: matrix, size = number of words in vocabulary x dimension of embedding space (number of features
    needed to describe a single word)
    :return: embedded matrix describing the sentence. size = #word in sentence x dimension of embedding space
    """
    embedded_sentences = np.zeros((len(sentence), embedding.shape[1]))
    zero_entries = 0
    for i, word in enumerate(sentence):
        try:
            index = vocabulary[word]
        except KeyError as e:
            zero_entries += 1
            continue
        embedded_sentences[i] = embedding[index]
    try:
        multiplier = len(sentence)/(len(sentence) - zero_entries)
    except ZeroDivisionError as e:
        multiplier = 0
    return embedded_sentences * multiplier


def extract_features(embedded_sentence):
    """
    Calculates feature vector
    :param embedded_sentence: np array of size: #words in sentence x dimension of embedding space
    :return: feature_vector: np array of size: dimension of embedding space containing means (features)
    """
    feature_vector = np.mean(embedded_sentence, axis=0)
    assert (feature_vector.shape[0] == embedded_sentence.shape[1]), "Shape of feature_vector is not correct"
    return feature_vector


def embed_dataset(dataset, vocabulary, embeddings):
    """Embeds the dataset."""
    embedded_dataset = []
    for sentence in dataset:
        embedded_dataset.append(embed_sentence(sentence, vocabulary, embeddings))
    return embedded_dataset


def compute_dataset_features(dataset, vocabulary, embeddings):
    """
    Computes the features for the entire dataset
    :param dataset: list of sentences
    :return: embedded dataset
    """
    dataset_features = np.zeros((len(dataset), np.shape(embeddings)[1]))
    for i, sentence in enumerate(dataset):
        dataset_features[i] = extract_features(embed_sentence(sentence, vocabulary, embeddings))
    return dataset_features


def train(model, dataloader_train, dataloader_val=None, num_epochs=10):
    """
    Main training procedure
    :return: losses and accuracies
    """
    iter_counter = 0
    current_loss = 0

    train_loss_history = []
    val_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []

    model.add_optimizer()

    for epoch in range(num_epochs):
        for x, labels in dataloader_train:
            if len(x) != dataloader_train.batch_size:
                continue

            x = Variable(x, requires_grad=False)
            labels = Variable(labels.type(torch.FloatTensor), requires_grad=False)

            model.optimizer.zero_grad()

            output = model(x)
            current_loss = model.loss(output, labels)

            current_loss.backward()
            model.optimizer.step()

            iter_counter += 1

        train_accuracy, train_loss = test(model, dataloader_train)
        print("epoch: {0}, train loss: {1}, train accuracy: {2}" .format(epoch, train_loss,
                                                             train_accuracy))
        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)

        if dataloader_val:
            val_accuracy, val_loss = test(model, dataloader_val)
            print("epoch: {0}, val loss: {1}, val accuracy: {2}".format(epoch, val_loss,
                                                                        val_accuracy))
            val_loss_history.append(val_loss)
            val_accuracy_history.append(val_accuracy)

    return train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history


def test(model, dataloader):
    """
    Evaluates the model on a dataset
    """
    dataset = Variable(dataloader.dataset.data_tensor, requires_grad=False)
    target = Variable(dataloader.dataset.target_tensor.type(torch.FloatTensor), requires_grad=False)
    output = model(dataset)
    accuracy = compute_accuracy(predict(output.data).numpy(), target.data.numpy())
    loss = model.loss(output, target)
    return accuracy, loss.data.numpy(),


def compute_accuracy(predictions, target):
    correct = predictions.ravel() == target.ravel()
    return np.sum(correct)/np.shape(target)[0]

def predict(output):
    return output > 0.5
