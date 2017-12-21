import torch
from torch import optim, nn
from torch.autograd import Variable

from src.rnn_modules.rnn_utils import sort_sequences, compute_accuracy, resume, save_checkpoint


def train(model, dataloader_train, train_lengths, config, dataloader_val=None,
          val_lengths=None):
    """

    :param model: Sentiment analysis predictor
    :param dataloader_train: iterable dataset
    :param train_lengths: lengths of the sentences in the dataset
    :param config: Config object with hyperparameters
    :param dataloader_val: iterable dataset for validation
    :param val_lengths: sentences' lengths for the validation set
    :return: history of losses and accuracies
    """
    start_epoch = 0
    train_loss_history = []
    val_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    loss = nn.CrossEntropyLoss()

    if config.resume:
        model, optimizer, start_epoch = resume(model, optimizer, config.directory + '/' +
                                    config.checkpoint_name)

    iter_num = 0
    for epoch in range(start_epoch, config.epochs_num):
        loss_value = 0
        accuracy = 0
        print("epoch : ", epoch)
        model.train()
        for index, batch in enumerate(dataloader_train):
            if len(batch[1]) != dataloader_train.batch_size:
                continue
            sequence, labels = batch
            # fetch the lengths
            sequence_lengths, _ = train_lengths[index*config.batch_size:(index+1)*config.batch_size]
            sequence, sequence_lengths, labels = sort_sequences(sequence, sequence_lengths, labels)

            labels_var = Variable(labels.type(torch.LongTensor), requires_grad=False)
            sequence_var = Variable(sequence.type(torch.LongTensor), requires_grad=False)

            optimizer.zero_grad()
            logits = model(sequence_var, sequence_lengths)
            loss_value = loss(logits, labels_var)

            loss_value.backward()
            # clip gradients
            nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, model.parameters()), 10.0)

            optimizer.step()

            accuracy = compute_accuracy(logits, labels_var)

            iter_num += 1
            if iter_num % 50 == 0:
                print("iter {0}, train loss: {1}, accuracy {2}" .format(
                    iter_num, loss_value.data.numpy(), accuracy))

        model.eval()
        # train_loss, train_accuracy = test(model, dataloader_train, train_lengths, config)
        # print("epoch: {0}, train loss: {1}, train accuracy: {2}".format(epoch,
        #                                                                 train_loss.data.numpy(),
        #                                                                 train_accuracy))
        train_loss_history.append(loss_value)
        train_accuracy_history.append(accuracy)

        if dataloader_val:
            val_loss, val_accuracy = test(model, dataloader_val, val_lengths, config)
            print("epoch: {0}, val loss: {1}, val accuracy: {2}".format(epoch, val_loss.data.numpy(),
                                                                        val_accuracy))
            val_loss_history.append(val_loss)
            val_accuracy_history.append(val_accuracy)

        save_checkpoint(model, optimizer, epoch, config.directory + '/rnn_' + str(epoch))

    return train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history


def test(model, dataloader, lengths, config):
    loss_value = 0
    loss = nn.CrossEntropyLoss()
    accuracy = 0
    num_batches = 0

    for index, batch in enumerate(dataloader):
        if len(batch[1]) != dataloader.batch_size:
            continue
        sequence, labels = batch
        sequence_lengths, _ = lengths[
                              index * config.batch_size:(index + 1) * config.batch_size]
        sequence, sequence_lengths, labels = sort_sequences(sequence, sequence_lengths, labels)

        labels_var = Variable(labels.type(torch.LongTensor), requires_grad=False)
        sequence_var = Variable(sequence.type(torch.LongTensor), requires_grad=False)

        logits = model(sequence_var, sequence_lengths)

        loss_value += loss(logits, labels_var)
        accuracy += compute_accuracy(logits, labels_var)
        num_batches += 1

    return loss_value/num_batches, accuracy/num_batches