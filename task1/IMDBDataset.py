from typing import TypeVar

import torch
from torch.utils.data import Dataset
import pandas as pd

T = TypeVar('T')


class IMDBDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.ngrams = torch.tensor([ngram for numeric in df['numeric'] for ngram in self.get_ngrams(numeric)])

    def __len__(self):
        return len(self.ngrams)

    def __getitem__(self, idx):
        return self.ngrams[idx]

    @staticmethod
    def get_ngrams(lst: list[T]) -> list[T]:
        if len(lst) < 2:
            return []
        return [lst[:i] for i in range(2, len(lst))]
