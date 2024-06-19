import os
import typing

import pandas as pd


def process_data(data_dir: str, save_dir: str, tokenizer: callable[str, list[str]]) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
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


def create_vocabulary(dfs: list[pd.DataFrame]) -> typing.Dict[str, int]:
    vocab = set()
    tokens: list[str]
    for df in dfs:
        for tokens in df['tokens']:
            vocab.update(tokens)

    vocabulary: typing.Dict[str, int] = dict()
    for i, token in enumerate(vocab):
        vocabulary[token] = i

    return vocabulary
