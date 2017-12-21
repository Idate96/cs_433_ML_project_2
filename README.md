# Project Text Sentiment Classification

The task of this competition is to predict if a tweet message used to contain a positive :) or negative :( smiley, by considering only the remaining text.
# First Steps

Clone the repository and create inside the following directories:
results, preprocess, embeddings, twitter-datasets

# Preprocessing and generation of embeddings
To create custom embedding run the following files:
```
build_vocab.sh
cut_vocab.sh
python3 pickle_vocab.py
python3 cooc.py
```

If you want to make use of the pretrained GloVe embeddings from the Stanford Group:
Download  glove.twitter.27B.zip from the following website:
https://nlp.stanford.edu/projects/glove/
and put it inside the folder embeddings

# How to run a simple NN
The nn models and utils are situated inside the src/rnn_modules folder.
Here you will find:

1. models.py contains the Linear, Single Layer and Double Layer NN model

2. train_utils.py contains the routine for training and testing the models

3. data_utils.py contains the general utils to process the data before passing them to the network

4. run.py contains the main procedure

To run one of the model:
```
python3 run.py
```
making sure the path is set to the root of library.
Inside run.py one can choose the spefic model to run and what kind of embeddings to load.


# How to run a RNN

The rnn models and utils are situated inside the src/rnn_modules folder.
Here you will find:

1. rnn_models.py contains an RNN and LSTM model

2. rnn_train.py contains the routine for training and testing the models

3. rnn_utils.py contains the general utils to process the data before passing them to the network

4. run.py contains the main procedure


To run one of the model:
```
python3 run.py
```
making sure the path is set to the root of library.
Inside run.py one can choose the spefic model to run and what kind of embeddings to load.
