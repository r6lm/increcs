#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# The BM routine can be run in two modes:
# 
# - hyperparameter tunning: use the validation cycle to find the most appropriate hyperparameters
# ```python
#     validation_start_period=<int>,
#     validation_end_period=<int>,
#     test_start_period=None,
#     test_end_period=None,
#     max_epochs=<int ge 30>,
# ```
# - test: obtain the performance metrics on the test cycle
# ```python
#     validation_start_period=None,
#     validation_end_period=None,
#     test_start_period=<int>,
#     test_end_period=<int>,
#     max_epochs=<int> # selected through validation
# ```

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np
import pandas as pd
import torch, torch.optim as optim, torch.nn as nn
import sys, os
from collections import defaultdict
sys.path.append("./..") # \todo: change for relative import
from dataset.ASMGMovieLens import ASMGMLDataModule
from utils.save import get_timestamp, save_as_json, get_path_from_re

from torch.utils.data import DataLoader
from MF.model import get_model
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# from pytorch_lightning.loggers import TensorBoardLogger

# filter warning for not setting validation set
import warnings
warnings.filterwarnings(
    "ignore", "Total length of `DataLoader` across ranks is zero.*")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device = }')

# get training regime experiment id
# timestamp = get_timestamp()


# # control flow parameters
# 

# In[ ]:


train_params = dict(
    input_path="../../data/preprocessed/ml_processed.csv",
    validation_start_period=11,
    validation_end_period=11,
    test_start_period=None,
    test_end_period=None,  # 25
    train_window=10,
    max_epochs=30,  # 20
    batch_size=1024,
    learning_rate=1e-3,  # 1e-2 is the ASMG MF implementation
    model_filename_stem='first_mf',
    seed=6202,
    save_model=False,
    save_result=True
)
model_params = dict(
    alias="MF",
    n_users=43183,
    n_items=51149,
    n_latents=8,
    l2_regularization_constant=1e-6,
)
train_params["model_checkpoint_dir"] = f'./../../model/{model_params["alias"]}/BM'


# In[ ]:


# initialize training components
torch.manual_seed(train_params["seed"])
model = get_model({**model_params, **train_params})#.to(device)
# loss_function = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=train_params["learning_rate"])

# progess log
progress_bar = TQDMProgressBar(refresh_rate=200)
early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=False, 
    min_delta=1e-4)

# initialize results container
res_dict = defaultdict(lambda: [])

# initialize dict of hyperparameters
train_hparams = {'learning_rate': train_params["learning_rate"],
            'l2_regularization_constant': model_params[
                "l2_regularization_constant"]}


# # validation

# In[ ]:


if train_params["validation_start_period"] is not None:

    # BU validation regime routine
    for val_period in range(
            train_params["validation_start_period"], train_params["validation_end_period"] + 1):

        # update periods
        train_window_begin = val_period - train_params["train_window"]
        train_window_end = val_period - 1
        print(
            f"train periods: {train_window_begin}-{train_window_end}",
            f"test period: {val_period}", sep="\n")

        # make checkpoint dir
        model_checkpoint_subdir = f'{train_params["model_checkpoint_dir"]}/' + (
            f'/V{val_period:02}')
        if not os.path.exists(model_checkpoint_subdir):
            os.makedirs(model_checkpoint_subdir)

        val_trainer = Trainer(
            accelerator="auto", devices=1 if torch.cuda.is_available() else 0,
            max_epochs=train_params["max_epochs"], reload_dataloaders_every_n_epochs=1,
            enable_checkpointing=train_params["save_model"],
            default_root_dir=model_checkpoint_subdir,
            enable_model_summary=(
                True if val_period == train_params["validation_start_period"]
                else False),
            callbacks=[
                early_stopping, progress_bar]
        )

        # load datsets
        train_dm = ASMGMLDataModule(
            train_params["input_path"], train_params["batch_size"],
            train_window_begin, train_window_end, period_val=val_period)

        # train
        val_trainer.fit(
            model, datamodule=train_dm)
        print(f"finished val_period {val_period}")

        # log hyperparameters
        val_trainer.logger.log_hyperparams({
            **train_hparams,
            'n_epochs': val_trainer.current_epoch - early_stopping.patience
        }, metrics=early_stopping.best_score)

        torch.manual_seed(train_params["seed"])
        model.reset_parameters()

else:
    print("skipped validation cycle")


# # test

# In[ ]:


if train_params["test_start_period"] is not None:
    
    # BU test routine
    for test_period in range(
        train_params["test_start_period"], train_params["test_end_period"] + 1):

        # update periods
        train_window_begin = test_period - train_params["train_window"]
        train_window_end = test_period - 1 
        print(
            f"train periods: {train_window_begin}-{train_window_end}", 
            f"test period: {test_period}", sep="\n")

        # make checkpoint dir
        model_checkpoint_subdir = f'{train_params["model_checkpoint_dir"]}' + (
            f'/T{test_period:02}' if train_params["save_model"] else "")
        if not os.path.exists(model_checkpoint_subdir):
            os.makedirs(model_checkpoint_subdir) 

        # model_checkpoint_path = f'{model_checkpoint_subdir}/' \
        #     f'{train_params["model_filename_stem"]}.pth'

        test_trainer = Trainer(
                accelerator="auto", devices=1 if torch.cuda.is_available() else 0,
                max_epochs=train_params["max_epochs"], reload_dataloaders_every_n_epochs=1,
                enable_checkpointing=train_params["save_model"],
                default_root_dir=model_checkpoint_subdir,
                logger=False, enable_model_summary=False,
                callbacks=[progress_bar]
            )

        # load datsets 
        train_dm = ASMGMLDataModule(
            train_params["input_path"], train_params["batch_size"], 
            train_window_begin, train_window_end)
        test_dm = ASMGMLDataModule(
            train_params["input_path"], train_params["batch_size"], test_period)

        # train
        test_trainer.fit(
            model, datamodule=train_dm)
        res_dict["testPeriod"].append(test_period)
        res_dict["trainLoss"].append(test_trainer.logged_metrics["train_loss"].item())
        print(f"finished test_period {test_period}")

        # test
        test_trainer.test(model, datamodule=test_dm)
        res_dict["testLoss"].append(test_trainer.logged_metrics["test_loss"].item())

        torch.manual_seed(train_params["seed"])
        model.reset_parameters()
        
    else:
        timestamp = get_timestamp()
        df_path = f"{model_checkpoint_subdir.replace(f'/T{test_period:02}', '')}/{timestamp}.csv"
        df_path

        # calculate and display average loss
        res_df = pd.DataFrame(res_dict)

        average_srs = res_df.mean()
        average_srs.at["testPeriod"] = "mean"
        print(average_srs)

        if train_params["save_result"]: 
            pd.concat((res_df, average_srs.to_frame().T), axis=0, ignore_index=True
            ).to_csv(df_path, index=False)
            print(f"saved results csv at: {os.path.abspath(df_path)}")

else:
    print("skipped test cycle")

