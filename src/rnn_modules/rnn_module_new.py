from torch import nn
import numpy as np
import torch
np.random.seed(5)
torch.manual_seed(40)
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
from src.nn_modules import data_utils
from src.rnn_modules.rnn_train import train
from src.rnn_modules.rnn_utils import sequence2tensor, generate_dataloader, \
    Config, save_data, create_submission_rnn, load_test_data, load_params


class RNNClassifier(nn.Module):
    def __init__(self, hidden_dim, config, embeddings, label=''):
        super().__init__()
        self.label = label
        self.config = config
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(config.vocab_size, embedding_dim=config.embedding_dim)
        # pretrained_weight is a numpy matrix of shape (num_embeddings, embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(embeddings))
        self.embedding.weight.requires_grad = True
        self.rnn = nn.RNN(config.embedding_dim, hidden_dim, num_layers=1, nonlinearity='tanh',
                          dropout=0.5, bidirectional=True)
        self.linear = nn.Linear(hidden_dim*2, 2)
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


def run():

    # get features and labels of tweets
    embedding_dim = 200
    print("Loading data ... ")
    embeddings, vocabulary, dataset, labels = load_params(embedding_dim,
                                                                     use_all_data=False)

    # embeddings = np.load('preprocess/embeddings.npy')
    # print(np.shape(embeddings))
    # embeddings = embeddings[:-2]
    # with open('preprocess' + '/' + 'vocab.pkl', 'rb') as f:
    #     vocab = pickle.load(f)
    # print(len(vocab))

    # hyperparameters
    config = Config(batch_size=500, embedding_dim=embedding_dim, vocab_size=len(vocabulary),
                    learning_rate=10**-3, epochs_num=5, directory='results/rnn_h256_e200_l3')
    print("Vectorizing sentences ...")
    dataset_tensor, lengths_tensor, parsed_targets = sequence2tensor(dataset, vocabulary, labels)

    dataloader_train, lengths_train, dataloader_test, lengths_test = generate_dataloader(
        dataset_tensor, lengths_tensor, parsed_targets, config.batch_size)

    model = RNNClassifier(hidden_dim=256, config=config, embeddings=embeddings,
                          label='rnn_h256_e200_l3')

    train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = train(
        model, dataloader_train, lengths_train, config, dataloader_test, lengths_test)

    save_data(config.epochs_num, train_loss_history, train_accuracy_history, val_loss_history,
                         val_accuracy_history, 'results/' + model.label)

    test_data = load_test_data()
    test_tensor, test_lengths, _ = sequence2tensor(test_data, vocabulary)
    create_submission_rnn(model, test_tensor, test_lengths, 'results/' + model.label)

if __name__ == '__main__':
    run()



