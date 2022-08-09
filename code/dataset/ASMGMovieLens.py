import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pytorch_lightning import LightningDataModule
from typing import Optional
import os


class ASMGMovieLens(Dataset):
    def __init__(self, input_path, period_start, period_end=None, x_cols=None):

        self.period_start = period_start
        self.period_end = period_end if period_end is not None else period_start
        self.x_cols = ["userId", "movieId",
                       "period"] if x_cols is None else x_cols

        # load data but do not keep in instance
        all_data_df = pd.read_csv(input_path)

        # select period
        data_df = all_data_df.loc[
            lambda df:df.period.between(self.period_start, self.period_end)]

        # select columns
        x = data_df.loc[:, self.x_cols].values
        y = data_df.loc[:, "label"].values

        self.x = torch.tensor(x, dtype=torch.int32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


if __name__ == "__main__":
    d = ASMGMovieLens(
        "/home/rodrigo/Dropbox/Documents/MSc/dissertation/increcs/data/preprocessed/ml_processed.csv",
        15, 24)
    d[31]
    print(d[0])


class ASMGMLDataModule(LightningDataModule):
    def __init__(
            self, input_path, batch_size: int, period_start, period_end: int = None,
            period_val: int = None):
        super().__init__()
        self.input_path = input_path
        self.batch_size = batch_size
        self.period_start = period_start
        self.period_end = period_end
        self.period_val = period_val

    def setup(self, stage: Optional[str] = "fit"):

        if stage == "fit" or stage is None:
            self.train_dataset = ASMGMovieLens(
                self.input_path, self.period_start, self.period_end)
            if self.period_val is not None:
                self.val_dataset = ASMGMovieLens(self.input_path, self.period_val)
            else:
                self.val_dataset = [] # this generates ´Userwarning:
                # Total length of `DataLoader` across ranks is zero....´ it can 
                # be filtered on scripts for less verbosity

        if stage == "test" or stage is None:
            self.test_dataset = ASMGMovieLens(self.input_path, self.period_start)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=os.cpu_count())

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=os.cpu_count())

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=os.cpu_count())
