import os
from typing import Tuple, Dict, TypeVar

import pandas as pd
import torch
from torch.nn.functional import pad
from torch.utils.data import TensorDataset, DataLoader


def process_data(data_dir: str, save_dir: str, tokenizer) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # extract the data from the files into a dictionary
    data = dict()
    for dataset in ["train", "test"]:
        curr = []
        print(f"Starting on: {dataset}")
        for part in ["pos", "neg"]:
            print(f"Part: {part}")
            directory = f"{data_dir}/{dataset}/{part}"
            for filename in os.listdir(directory):
                id, rating = filename.replace(".txt", "").split('_')
                with open(os.path.join(directory, filename), mode="r", encoding="utf-8") as f:
                    text = f.read()
                curr.append({"dataset": dataset, "part": part, "id": id, "rating": rating, "text": text})
        data[dataset] = curr
    # create dataframes from the sets
    df_train = pd.DataFrame(data["train"])
    df_test = pd.DataFrame(data["test"])
    # tokenize the sentences
    df_train["tokens"] = df_train["text"].map(tokenizer)
    df_test["tokens"] = df_test["text"].map(tokenizer)
    # save the end result to csv files
    df_train.to_csv(f"{save_dir}/train.csv")
    df_test.to_csv(f"{save_dir}/test.csv")
    # return the 2 dataframes
    return df_train, df_test


def create_vocabulary(dfs: list[pd.DataFrame]) -> Dict[str, int]:
    vocab = set()
    tokens: list[str]
    for df in dfs:
        for tokens in df['tokens']:
            vocab.update(tokens)

    vocabulary: Dict[str, int] = dict()
    for i, token in enumerate(vocab):
        vocabulary[token] = i + 1

    return vocabulary


T = TypeVar('T')


def get_ngrams(lst: list[T], max_size: int) -> list[list[T]]:
    if len(lst) < 2 or max_size < 2:
        return []

    return [lst[i:i + max_size] for i in range(min(0, len(lst) - max_size))] + [lst[:i] for i in
                                                                                range(2, min(len(lst), max_size))]


def create_dataset(ngrams: list[list[T]]) -> Tuple[TensorDataset, int]:
    X = [ngram[:-1] for ngram in ngrams]
    max_length_x = max(len(x) for x in X)
    padded_X = torch.stack([pad(torch.tensor(x), (max_length_x - len(x), 0), value=0) for x in X])
    y = torch.tensor([ngram[-1] for ngram in ngrams])
    dataset = TensorDataset(padded_X, y)
    return dataset, max_length_x
