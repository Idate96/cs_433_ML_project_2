import pickle
import numpy as np


def get_train_arrays_keras(file, train_arrays, train_labels, positive=1, counter=0):
    f = open(file, 'r')
    for line in f:
        idxs_tweet = [vocab.get(t, -1) for t in line.strip().split()]  # get idx of each tweet in vocabulary
        idxs_tweet = [t for t in idxs_tweet if t >= 0]  # performs a check for words not present in vocabulary
        idxs2vects = [embeddings[i] for i in idxs_tweet]  # trasforms idx vocabulary in wordvect

        if len(idxs_tweet) > 0:
            sum_vect_avg = sum(idxs2vects) / len(idxs_tweet)  # gives array of full tweet if any
            train_arrays[counter] = sum_vect_avg
            train_labels[counter] = positive
            counter += 1
    f.close()
    return train_arrays, train_labels, counter

def get_test_arrays_keras(file, test_arrays, counter = 0):
    f = open(file, 'r')
    for line in f:
        idxs_tweet = [vocab.get(t, -1) for t in line.strip().split()]  # get idx of each tweet in vocabulary
        idxs_tweet = [t for t in idxs_tweet if t >= 0]  # performs a check for words not present in vocabulary
        idxs2vects = [embeddings[i] for i in idxs_tweet]  # trasforms idx vocabulary in wordvect

        if len(idxs_tweet) > 0:
            sum_vect_avg = sum(idxs2vects) / len(idxs_tweet)  # gives array of full tweet if any
            test_arrays[counter] = sum_vect_avg

        else:
            test_arrays[counter] = counter

        counter += 1
    f.close()
    return test_arrays, counter

def count_lines(file):
    f = open(file, 'r')
    i = 0
    maximum = 0
    for









folder = './build-vocab/'
with open(folder+'vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

embeddings = np.load(folder+'embeddings.npy')

#100000 tweets each
train_arrays = np.zeros((200000, 300))
train_labels = np.zeros(200000)

folder_data='./twitter-datasets-pp/'

"""
positive_test = folder_data+'train_pos.txt'
train_arrays, train_labels, counter = get_train_arrays(positive_test, train_arrays, train_labels)
negative_test = folder_data+'train_neg.txt'
train_arrays, train_labels, counter = get_train_arrays(negative_test, train_arrays, train_labels, -1, counter)

train_arrays = train_arrays[:counter]
train_labels = train_labels[:counter]
#print(train_labels)
#print(train_arrays)
#print(counter)
with open(folder_data+'train_arrays.pkl', 'wb') as f:
    pickle.dump(train_arrays, f, pickle.HIGHEST_PROTOCOL)

with open(folder_data+'train_labels.pkl', 'wb') as f:
    pickle.dump(train_labels, f, pickle.HIGHEST_PROTOCOL)


test = folder_data+'test_data_new.txt'
test_arrays = np.zeros((10000, 300))
test_arrays, counter = get_test_arrays(test, test_arrays)
test_arrays = test_arrays[:counter]
print(test_arrays.shape)
#print(test_arrays)
#print(counter)
with open(folder_data+'test_arrays.pkl', 'wb') as f:
    pickle.dump(test_arrays, f, pickle.HIGHEST_PROTOCOL)
"""
