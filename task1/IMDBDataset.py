import torch
from torch.utils.data import Dataset
import pandas as pd


class IMDBDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.tokens = torch.tensor(df['tokens'])

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]
