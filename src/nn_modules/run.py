import sys, os
sys.path.insert(1, os.path.join(sys.path[0], "../../"))
from src.nn_modules.train_utils import *
from src.nn_modules.models import *
import pickle
from src.nn_modules.models import LinearBCEModel
from src.nn_modules.data_utils import *


def main():
    print("staring")
    # hyperparameters
    embedding_dim = 200
    batch_size = 500
    learning_rate = 10**-3
    epochs_num = 0
    embeddings_name = 'embeddings_' + str(embedding_dim)


    # get features and labels of tweets
    #embeddings, vocabulary, dataset, labels = data_utils.load_params(embeddings_name=embeddings_name, use_all_data=True)

    embeddings = np.load('embeddings/stanford_v1/final_embedding_200d.npy')
    with open('embeddings//stanford_v1/vocab_final_72153.pkl', 'rb') as file:
        vocabulary = pickle.load(file)
    with open('twitter-datasets-pp/train_pos_full.txt', encoding="utf-8") as file:
        positive = file.readlines()
    file.close()
    with open('twitter-datasets-pp/train_neg_full.txt', encoding="utf-8") as file:
        negative = file.readlines()
    file.close()

    dataset, labels = data_utils.load_train_dataset(positive, negative)
    dataset_features = compute_dataset_features(dataset, vocabulary, embeddings)
    dataloader_train, dataloader_val = data_utils.generate_dataloader(dataset_features, labels, batch_size)

    # learning algorithm

    print("preparing model...")
    model = BCEModel(embedding_dim, learning_rate, l2_reg=5*10**-2)
    print("model loaded. Start training ...")

    train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = train(model, dataloader_train, dataloader_val, epochs_num)
    identification = 'glove_old_pp_BEST_10epoch'
    # plots
    data_utils.plot_convergence(epochs_num, train_loss_history, val_loss_history, model.name+identification)
    data_utils.plot_accuracy(epochs_num, train_accuracy_history, val_accuracy_history, model.name+identification)

    # save data
    data_utils.save_data(epochs_num, train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history,
                         'results/' + model.name+identification)

    # create submission
    test_data = data_utils.load_test_data()
    test_data_features = compute_dataset_features(test_data, vocabulary, embeddings)
    data_utils.create_csv_submission(model, test_data_features, model.name+identification)


if __name__ == "__main__":
    main()
