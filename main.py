import time
from pathlib import Path

import nltk
import pandas as pd
import torch
from nltk.tokenize import word_tokenize
from torch import optim, nn, Tensor, IntTensor
from torch.nn.modules.loss import _Loss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Optimizer
from torch.types import Device
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from task1.MyRnn import MyRnn
from task1.process_data import process_data, create_vocabulary, create_dataset


def does_file_exist(file: str) -> bool:
    return Path(file).is_file()


def train_my_rnn(model: MyRnn, train_dataloader: DataLoader, validation_dataloader: DataLoader,
                 loss_function: _Loss, optimizer: Optimizer, epochs: int,
                 device: Device) -> tuple[MyRnn, list[float], list[float], list[float], list[float]]:
    print("--------------------------------------------------")
    # initialize result holders
    train_accuracies = []
    train_losses = []
    validation_accuracies = []
    validation_losses = []
    # set model to training mode for the duration of the training
    for epoch in range(epochs):
        start = time.time()
        model.train()
        batch_lengths: IntTensor
        batch_data: Tensor
        batch_train_losses = []
        epoch_train_correct_predictions = 0
        epoch_train_total_predictions = 0
        for batch_lengths, batch_data in tqdm(train_dataloader, unit="item", unit_scale=train_dataloader.batch_size):
            # zero the gradient each batch
            optimizer.zero_grad()
            # fix data for the model
            batch_data = batch_data.to(device)
            # get the predictions
            outputs: Tensor
            outputs = model((batch_lengths, batch_data), model.create_hidden(train_dataloader.batch_size, device))
            # calculate the loss
            packed = pack_padded_sequence(batch_data, batch_lengths, batch_first=True, enforce_sorted=False)
            unpacked, _ = pad_packed_sequence(packed, batch_first=True)
            loss: Tensor = loss_function(outputs.transpose(1, 2), unpacked[:, 1:-1])
            batch_train_losses.append(loss.item())
            # propagate the loss backwards
            loss.backward()
            # update the weights
            optimizer.step()
            # add the correct predictions
            epoch_train_correct_predictions += (outputs.argmax(dim=2) == unpacked[:, 1:-1]).sum().item()
            # add the total predictions
            epoch_train_total_predictions += batch_lengths.sum().item()

        # log the epoch time
        epoch_time = time.time()
        # add the final loss in the epoch to the list
        train_losses.append(sum(batch_train_losses) / len(batch_train_losses))
        # add the final accuracy in the epoch to the list
        train_accuracies.append(epoch_train_correct_predictions / epoch_train_total_predictions)
        # calculate the accuracy and loss on the validation set
        batch_validation_losses = []
        validation_correct_predictions = 0
        validation_total_predictions = 0
        model.eval()
        with torch.no_grad():
            for batch_lengths, batch_data in tqdm(validation_dataloader, unit="item", unit_scale=validation_dataloader.batch_size):
                # fix data for the model
                batch_data = batch_data.to(device)
                # get the predictions
                packed = pack_padded_sequence(batch_data, batch_lengths, batch_first=True, enforce_sorted=False)
                unpacked, _ = pad_packed_sequence(packed, batch_first=True)
                outputs: Tensor
                outputs = model((batch_lengths, batch_data), model.create_hidden(train_dataloader.batch_size, device))
                # calculate the loss
                loss: Tensor = loss_function(outputs.transpose(1, 2), unpacked[:, 1:-1])
                batch_train_losses.append(loss.item())
                # add the batch size to the total predictions
                validation_total_predictions += batch_lengths.sum().item()
                # add the loss to the list
                batch_validation_losses.append(loss.item())
                # add the correct predictions
                validation_correct_predictions += (outputs.argmax(dim=2) == unpacked[:, 1:-1]).sum().item()
        # get the accuracy of the epoch and put it in the accuracies list
        validation_accuracies.append(validation_correct_predictions / validation_total_predictions)
        # get the loss of the epoch and put it in the losses list
        validation_losses.append(sum(batch_validation_losses) / len(batch_validation_losses))
        # print the results
        print(f"Epoch: {epoch + 1}")
        print(f"Train Accuracy: {train_accuracies[-1]}, Train Loss: {train_losses[-1]}")
        print(f"Validation Accuracy: {validation_accuracies[-1]}, Validation Loss: {validation_losses[-1]}")
        print(f"Time: {int(epoch_time - start)} (+{int(time.time() - epoch_time)}) seconds")
        print("--------------------------------------------------")

    return model, train_accuracies, train_losses, validation_accuracies, validation_losses


if __name__ == '__main__':
    # constants
    data_dir = "data/original/aclImdb"
    save_dir = "data/processed"
    # check if the files exist
    train_data_file = Path(f"{save_dir}/train.csv")
    test_data_file = Path(f"{save_dir}/test.csv")
    found = train_data_file.is_file() and test_data_file.is_file()
    if found:
        # read files
        df_train, df_test = pd.read_csv(train_data_file), pd.read_csv(test_data_file)
    else:
        # process the data into dataframes
        nltk.download('punkt')
        df_train, df_test = process_data(data_dir=data_dir, save_dir=save_dir,
                                         tokenizer=word_tokenize)
    # create the vocabulary object
    print("Creating vocabulary")
    vocabulary = create_vocabulary([df_train, df_test], ["<PAD>", "<UNK>"])
    print(f"vocabulary size: {len(vocabulary)}")
    print("Finished creating vocabulary")
    # transform the tokens in the dataframes to their numeric equivalents
    print("Transforming tokens to numeric")
    df_train['numeric'] = df_train['tokens'].apply(lambda tokens: [vocabulary[token] for token in tokens])
    df_test['numeric'] = df_test['tokens'].apply(lambda tokens: [vocabulary[token] for token in tokens])
    print("Finished transforming tokens to numeric")
    # save with numeric versions of tokens
    if not found:
        df_train.to_csv(f"{save_dir}/train.csv")
        df_test.to_csv(f"{save_dir}/test.csv")
    # create the dataset and loader
    print("Getting numeric tokens")
    numeric_train: list[list[int]] = df_train['numeric'].to_list()
    numeric_test: list[list[int]] = df_test['numeric'].to_list()
    print("Finished getting numeric tokens")
    # create the tensor datasets
    print("Creating datasets")
    batch_size = 16
    if does_file_exist("test_dataset.pth"):
        test_dataset = torch.load("test_dataset.pth")
    else:
        test_dataset = create_dataset(numeric_test)
        torch.save(test_dataset, "test_dataset.pth")
    if does_file_exist("train_dataset.pth") and does_file_exist("validation_dataset.pth"):
        train_dataset = torch.load("train_dataset.pth")
        validation_dataset = torch.load("validation_dataset.pth")
    else:
        if does_file_exist("dataset.pth"):
            dataset = torch.load("dataset.pth")
        else:
            dataset = create_dataset(numeric_train)
            torch.save(dataset, "dataset.pth")
        train_dataset, validation_dataset = random_split(dataset, [0.8, 0.2], torch.Generator().manual_seed(42))
        torch.save(train_dataset, "train_dataset.pth")
        torch.save(validation_dataset, "validation_dataset.pth")
    print("Finished creating datasets")
    # delete the dataframes to save some memory
    del df_train
    del df_test
    # initialize the dataset and loader
    print("Creating dataloaders")
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    print(f"Train dataloader size: {len(train_dataloader)}")
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
    print(f"Validation dataloader size: {len(validation_dataloader)}")
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Test dataloader size: {len(test_dataloader)}")
    print("Finished creating dataloaders")
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
    epochs = 50
    optimizer = optim.Adam(model.parameters(), lr=0.0009)
    loss_function = nn.CrossEntropyLoss()
    model, train_accuracies, train_losses, validation_accuracies, validation_losses \
        = train_my_rnn(model, train_dataloader, validation_dataloader, loss_function, optimizer, epochs, device)
    torch.save(model.state_dict(), "model.pth")
    print("Finished training model")
