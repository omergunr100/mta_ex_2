import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import torch
from nltk.tokenize import word_tokenize
from torch import optim, nn
from torch.utils.data import DataLoader, random_split

from task1.evaluate import evaluate_model
from task1.model.MyRnn import MyRnn
from task1.process_data import process_data, create_vocabulary, create_dataset
from task1.train import train_my_rnn


def does_file_exist(file: str) -> bool:
    return Path(file).is_file()


if __name__ == '__main__':
    # constants
    data_dir = "task1/data/original/aclImdb"
    save_dir = "task1/data/processed"
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
    vocab_file = Path(f"{save_dir}/vocabulary.json")
    if vocab_file.is_file():
        with open(vocab_file, "r", encoding="utf-8") as vocabulary_file:
            vocabulary = json.load(vocabulary_file)
    else:
        vocabulary = create_vocabulary([df_train, df_test], ["<PAD>", "<UNK>"])
        with open(f"{save_dir}/vocabulary.json", "w", encoding="utf-8") as vocabulary_file:
            json.dump(vocabulary, vocabulary_file)
    print(f"vocabulary size: {len(vocabulary)}")
    print("Finished creating vocabulary")
    # transform the tokens in the dataframes to their numeric equivalents
    print("Transforming tokens to numeric")
    df_train['numeric'] = df_train['tokens'].apply(lambda tokens: [vocabulary.get(token, vocabulary["<UNK>"]) for token in tokens])
    df_test['numeric'] = df_test['tokens'].apply(lambda tokens: [vocabulary.get(token, vocabulary["<UNK>"]) for token in tokens])
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
    if does_file_exist(f"{save_dir}/test_dataset.pth"):
        test_dataset = torch.load(f"{save_dir}/test_dataset.pth")
    else:
        test_dataset = create_dataset(numeric_test)
        torch.save(test_dataset, f"{save_dir}/test_dataset.pth")
    if does_file_exist(f"{save_dir}/train_dataset.pth") and does_file_exist(
            f"{save_dir}/validation_dataset.pth"):
        train_dataset = torch.load(f"{save_dir}/train_dataset.pth")
        validation_dataset = torch.load(f"{save_dir}/validation_dataset.pth")
    else:
        if does_file_exist(f"{save_dir}/dataset.pth"):
            dataset = torch.load(f"{save_dir}/dataset.pth")
        else:
            dataset = create_dataset(numeric_train)
            torch.save(dataset, f"{save_dir}/dataset.pth")
        train_dataset, validation_dataset = random_split(dataset, [0.8, 0.2], torch.Generator().manual_seed(42))
        torch.save(train_dataset, f"{save_dir}/train_dataset.pth")
        torch.save(validation_dataset, f"{save_dir}/validation_dataset.pth")
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
    model_exists = does_file_exist("task1/model/model.pth")
    if model_exists:
        model = torch.load("task1/model/model.pth")
    else:
        model = MyRnn(vocab_dim=vocab_size, output_size=vocab_size, embedding_dim=100, n_layers=2, dropout=0.5,
                      hidden_size=100)
    print("Finished creating model")
    # move model to gpu if available
    print("Moving model to device")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Finished moving model to device: ", device)
    # if the model doesn't exist generate a new one
    if not model_exists:
        # train the model
        print("Training model")
        epochs = 8
        optimizer = optim.Adam(model.parameters(), lr=0.0009)
        loss_function = nn.CrossEntropyLoss()
        model, train_accuracies, train_losses, validation_accuracies, validation_losses \
            = train_my_rnn(model, train_dataloader, validation_dataloader, loss_function, optimizer, epochs, device)
        torch.save(model, "task1/model/model.pth")
        print("Finished training model")
        # plot the results
        print("Plotting results")
        plt.plot(train_accuracies, label="Train")
        plt.plot(validation_accuracies, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(["Train", "Validation"], loc="upper left")
        plt.title("Accuracy")
        plt.savefig("task1/accuracy_graph.png")
        plt.show()
        plt.plot(train_losses, label="Train")
        plt.plot(validation_losses, label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(["Train", "Validation"], loc="upper left")
        plt.title("Loss")
        plt.savefig("task1/loss_graph.png")
        plt.show()
        print("Finished plotting results")
    # test the model
    print("Testing model")
    acc, loss, perplexity = evaluate_model(model, test_dataloader, device)
    print(f"Test Accuracy: {acc}, Loss: {loss}, Perplexity: {perplexity}")
    print("Finished testing model")
