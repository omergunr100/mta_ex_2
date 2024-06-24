from pathlib import Path

import torch
from torch.utils.data import random_split

if __name__ == "__main__":
    # constants
    data_dir = "task1/data/processed"
    save_dir = "task2/data/processed"
    # create the datasets and loaders
    print("Creating datasets and loaders")
    train_file = Path(f"{save_dir}/train_dataset.pth")
    validation_file = Path(f"{save_dir}/validation_dataset.pth")
    test_file = Path(f"{save_dir}/test_dataset.pth")
    if train_file.is_file() and validation_file.is_file() and test_file.is_file():
        train_dataset = torch.load(train_file)
        validation_dataset = torch.load(validation_file)
        test_dataset = torch.load(test_file)
    else:
        dataset = torch.load(f"{data_dir}/dataset.pth")
        test_dataset = torch.load(f"{data_dir}/test_dataset.pth")
        train_dataset, validation_dataset = random_split(dataset, [0.2, 0.8], torch.Generator().manual_seed(42))
    print("Finished creating datasets and loaders")
