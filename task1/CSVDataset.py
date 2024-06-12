from torch.utils.data import Dataset
import pandas as pd


class CSVDataset(Dataset):
    def __init__(self, sample_file: str, length_override: int = 0):
        self.data = pd.read_csv(sample_file)
        self.length = length_override

    def __len__(self):
        return len(self.data) if self.length <= 0 else min(self.length, len(self.data))

    def __getitem__(self, idx):
        return self.data.iloc[idx]
