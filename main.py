import time

import torch
from nltk.tokenize import word_tokenize
from torch import optim, nn
from torch.nn.functional import pad
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from task1.MyRnn import MyRnn
from task1.process_data import process_data, create_vocabulary, get_ngrams


def train_my_rnn(model: MyRnn, dataloader: DataLoader, loss_function: _Loss, optimizer: Optimizer, epochs: int,
                 device: torch.device) -> tuple[MyRnn, list[float], list[float]]:
    # initialize result holders
    accuracies = []
    losses = []
    # set model to training mode for the duration of the training
    for epoch in range(epochs):
        start = time.time()
        model.train()
        batch_X: torch.Tensor
        batch_y: torch.Tensor
        for batch_X, batch_y in dataloader:
            # move the data to the selected device
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            # zero the gradient each batch
            optimizer.zero_grad()
            # go over all ngrams in the batch
            outputs = model(batch_X)
            # calculate the loss
            loss: torch.Tensor = loss_function(outputs, batch_y)
            # propagate the loss backwards
            b = loss.backward()
            # update the weights
            optimizer.step()

        # log the epoch time
        epoch_time = time.time()
        # add the final loss in the epoch to the list
        losses.append(loss.item())
        # calculate the accuracy and loss for the epoch
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            batch_X: torch.Tensor
            batch_y: torch.Tensor
            for batch_X, batch_y in dataloader:
                # move the data to the selected device
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                # add the batch size to the total predictions
                total_predictions += len(batch_y)
                # get the predictions
                predictions = model(batch_X).argmax(dim=1)
                # add the correct predictions
                correct_predictions += (predictions == batch_y).sum().item()
        # get the accuracy of the epoch and put it in the accuracies list
        accuracies.append(correct_predictions / total_predictions)
        # print the results
        print(f"Epoch: {epoch + 1}, Accuracy: {accuracies[-1]}, Loss: {losses[-1]}, Time: {int(epoch_time - start)} (+{int(time.time() - epoch_time)}) seconds")

    return model, accuracies, losses


if __name__ == '__main__':
    # constants
    data_dir = "data/original/aclImdb"
    save_dir = "data/processed"
    # process the data into dataframes
    df_train, df_test = process_data(data_dir=data_dir, save_dir=save_dir,
                                     tokenizer=word_tokenize)
    # create the vocabulary object
    vocabulary = create_vocabulary([df_train, df_test])
    # transform the tokens in the dataframes to their numeric equivalents
    df_train['numeric'] = df_train['tokens'].apply(lambda tokens: [vocabulary[token] for token in tokens])
    df_test['numeric'] = df_test['tokens'].apply(lambda tokens: [vocabulary[token] for token in tokens])
    # save with numeric versions of tokens
    df_train.to_csv(f"{save_dir}/train.csv")
    df_test.to_csv(f"{save_dir}/test.csv")
    # create the dataset and loader
    print("Creating ngrams")
    numeric: list[int]
    ngrams = [ngram for numeric in df_train['numeric'] for ngram in get_ngrams(numeric, 8)]
    print("Finished creating ngrams")
    # get and modify X to be ready for training
    print("Creating X")
    X = [ngram[:-1] for ngram in ngrams]
    max_length_x = max(len(x) for x in X)
    padded_X = torch.stack([pad(torch.tensor(x), (max_length_x - len(x), 0), value=0) for x in X])
    print("Finished creating X")
    # get and modify y to be ready for training
    print("Creating y")
    y = torch.tensor([ngram[-1] for ngram in ngrams])
    print("Finished creating y")
    # delete the dataframes to save some memory
    del df_train
    del df_test
    # initialize the dataset and loader
    print("Creating dataset and loader")
    batch_size = 64
    train_dataset = TensorDataset(padded_X, y)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    print("Finished creating dataset and loader")
    # delete the dataframes to save memory
    # create instance of my rnn
    print("Creating model")
    vocab_size = len(vocabulary)
    model = MyRnn(vocab_dim=vocab_size, output_size=vocab_size, embedding_dim=100, n_layers=2, dropout=0.5,
                  hidden_size=100)
    print("Finished creating model")
    # move model to gpu if available
    print("Moving model to device")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Finished moving model to device: ", device)
    # train the model
    print("Training model")
    epochs = 20
    optimizer = optim.Adam(model.parameters(), lr=0.0009)
    loss_function = nn.CrossEntropyLoss()
    model, accuracies, losses = train_my_rnn(model, train_dataloader, loss_function, optimizer, epochs, device)
    torch.save(model.state_dict(), "model.pth")
    print("Finished training model")
