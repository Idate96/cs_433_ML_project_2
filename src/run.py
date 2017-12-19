from src.nn_modules.sentiment import *
from src.nn_modules.models import *


def main():
    # hyperparameters
    embedding_dim = 50
    batch_size = 500
    learning_rate = 10**-2
    epochs_num = 20

    # get features and labels of tweets
    embeddings, vocabulary, dataset, labels = data_utils.load_params(embedding_dim,
                                                                     use_all_data=True)
    dataset_features = compute_dataset_features(dataset, vocabulary, embeddings)
    dataloader_train, dataloader_val = data_utils.generate_dataloader(dataset_features, labels,
                                                                      batch_size)

    # learning algorithm

    print("preparing model...")
    model = BCEModel(embedding_dim, learning_rate, l2_reg=10**-3, number_neurons=256)
    print("model loaded. Start training data...")
    train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = train(model, dataloader_train,
                                                                                           dataloader_val, epochs_num)

    # plots
    data_utils.plot_convergence(epochs_num, train_loss_history, val_loss_history, model.name)
    data_utils.plot_accuracy(epochs_num, train_accuracy_history, val_accuracy_history, model.name)

    # save data
    data_utils.save_data(epochs_num, train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history,
                         'results/' + model.name)

    # create submission
    test_data = data_utils.load_test_data()
    test_data_features = compute_dataset_features(test_data, vocabulary, embeddings)
    data_utils.create_csv_submission(model, test_data_features, model.name)


if __name__ == "__main__":
    main()
