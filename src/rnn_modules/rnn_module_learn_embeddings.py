from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.nn_modules import data_utils
from src.rnn_modules.rnn_train import train
from src.rnn_modules.rnn_utils import sequence2tensor, generate_dataloader, Config


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


def run():
    # hyperparameters
    config = Config(batch_size=20, embedding_dim=20, vocab_size=21161, learning_rate=10 ** -3, \
                    epochs_num=20)
    # get features and labels of tweets
    print("Loading data ... ")
    embeddings, vocabulary, dataset, labels = data_utils.load_params(config.embedding_dim,
                                                                     use_all_data=False)
    print("Vectorizing sentences ...")
    dataset_tensor, lengths_tensor, parsed_targets = sequence2tensor(dataset, vocabulary, labels)

    dataloader_train, lengths_train, dataloader_test, lengths_test = generate_dataloader(
        dataset_tensor, lengths_tensor, parsed_targets, config.batch_size)

    model = RNNClassifier(hidden_dim=20, config=config)

    train(model, dataloader_train, lengths_train, config, dataloader_test, lengths_test)

if __name__ == '__main__':
    run()



