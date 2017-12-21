import sys, os
sys.path.insert(1, os.path.join(sys.path[0], "../../"))
from src.rnn_modules.rnn_models import RNNClassifier, LSTMClassifier
from src.rnn_modules.rnn_train import train
from src.rnn_modules.rnn_utils import load_params, Config, sequence2tensor, generate_dataloader, \
    save_data, load_test_data, create_submission_rnn, load_train_dataset
import numpy as np
import pickle


def run():
    # get features and labels of tweets
    embedding_dim = 25
    custom_embeddings = False
    print("Loading data ... ")
    if custom_embeddings:
        embeddings, vocabulary, dataset, labels = load_params(embedding_dim,
                                                                     use_all_data=True)
    else:
        embeddings = np.load('embeddings/stanford_25/final_embedding_25d.npy')
        with open('embeddings/stanford_25/vocab_final_72153_25d.pkl', 'rb') as file:
            vocabulary = pickle.load(file)
        dataset, labels = load_train_dataset(use_all_data=True)

    # hyperparameters
    config = Config(batch_size=500, embedding_dim=embedding_dim, vocab_size=len(vocabulary),
                    learning_rate=10**-3, epochs_num=5,
                    directory='results/lstm_h20_e20_all',
                    resume=True, checkpoint_name='rnn_0')

    print("Vectorizing sentences ...")
    dataset_tensor, lengths_tensor, parsed_targets = sequence2tensor(dataset, vocabulary, labels)
    # generate the dataloader (iterable)
    dataloader_train, lengths_train, dataloader_test, lengths_test = generate_dataloader(
        dataset_tensor, lengths_tensor, parsed_targets, config.batch_size)

    model = LSTMClassifier(hidden_dim=25, config=config, embeddings=embeddings,
                          label='lstm_h20_e20_all')
    # log values
    train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = train(
        model, dataloader_train, lengths_train, config, dataloader_test, lengths_test)

    save_data(config.epochs_num, train_loss_history, train_accuracy_history, val_loss_history,
                         val_accuracy_history, 'results/' + model.label)
    # generate submission
    test_data = load_test_data()
    test_tensor, test_lengths, _ = sequence2tensor(test_data, vocabulary)
    create_submission_rnn(model, test_tensor, test_lengths, 'results/' + model.label)


if __name__ == '__main__':
    run()