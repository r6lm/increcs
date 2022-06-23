import torch
from torch.utils.data import Dataset
import pandas as pd


class ASMGMovieLens(Dataset):
  def __init__(self, input_path, period_start, period_end=None):
    
    self.x_train = None
    self.y_train = None
    self.all_data_df = pd.read_csv(input_path)
    self.set_period(period_start, period_end)

  def set_period(self, period_start, period_end=None): 
    self.period_start = period_start
    self.period_end = period_end if period_end is not None else period_start
    if hasattr(self, "all_data_df"):
        self.update_xy()

  def update_xy(self):
    data_df = self.all_data_df.loc[
      lambda df:df.period.between(self.period_start, self.period_end)]

    x = data_df.loc[:, ["userId", "movieId", "period"]].values
    y = data_df.loc[:, "label"].values

    self.x_train = torch.tensor(x, dtype=torch.float32)
    self.y_train = torch.tensor(y, dtype=torch.float32)

  def __len__(self):
    return len(self.y_train)
   
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]

if __name__ == "__main__":
    d = ASMGMovieLens(
        "/home/rodrigo/Dropbox/Documents/MSc/dissertation/increcs/data/preprocessed/ml_processed.csv", 
        15, 24)
    d.set_period(31)
    print(d[0])