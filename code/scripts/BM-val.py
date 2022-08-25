#!/usr/bin/env python
# coding: utf-8

# In[1]:

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

import argparse

parser = argparse.ArgumentParser(
    description='Run validation of BM training scheme.')

parser.add_argument('--l2-reg', help='l2 regularization constant')

args = parser.parse_args()
print(args.l2_reg)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print(f'{device = }')

# # get model checkpoint path
# # timestamp = get_timestamp()


# # # control flow parameters
# # 

# # In[2]:


# train_params = dict(
#     input_path="../../data/preprocessed/ml_processed.csv",
#     validation_start_period=24,
#     validation_end_period=24,
#     test_start_period=25,
#     test_end_period=25, #26
#     train_window=10,
#     max_epochs=30,
#     batch_size = 1024,
#     learning_rate = 1e-3, # 1e-2 is the ASMG MF implementation
#     l2_regularization_constant = 1e-2,
#     model_filename_stem = 'first_mf',
#     seed=6202,
#     save_model=False,
#     save_result=True
# )
# model_params = dict(
#     alias="MF",
#     n_users=43183,
#     n_items=51149,
#     n_latents=8
# )
# train_params["model_checkpoint_dir"] = f'./../../model/{model_params["alias"]}/BM'


# # In[3]:


# # initialize training components
# torch.manual_seed(train_params["seed"])
# model = get_model({**model_params, **train_params})#.to(device)
# # loss_function = nn.BCELoss()
# # optimizer = optim.Adam(model.parameters(), lr=train_params["learning_rate"])

# # progess log
# progress_bar = TQDMProgressBar(refresh_rate=200)
# early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=False, 
#     min_delta=1e-4)

# # initialize results container
# res_dict = defaultdict(lambda: [])

# # initialize dict of hyperparameters
# train_hparams = {'learning_rate': train_params["learning_rate"],
#             'l2_regularization_constant': train_params[
#                 "l2_regularization_constant"]}


# # # train

# # In[4]:



# # BU validation regime routine
# if train_params["validation_start_period"] is not None:

#     for val_period in range(
#             train_params["validation_start_period"], train_params["validation_end_period"] + 1):

#         # update periods
#         train_window_begin = val_period - train_params["train_window"]
#         train_window_end = val_period - 1
#         print(
#             f"train periods: {train_window_begin}-{train_window_end}",
#             f"test period: {val_period}", sep="\n")

#         # make checkpoint dir
#         model_checkpoint_subdir = f'{train_params["model_checkpoint_dir"]}/' + (
#             f'/V{val_period:02}')
#         if not os.path.exists(model_checkpoint_subdir):
#             os.makedirs(model_checkpoint_subdir)

#         val_trainer = Trainer(
#             accelerator="auto", devices=1 if torch.cuda.is_available() else 0,
#             max_epochs=train_params["max_epochs"], reload_dataloaders_every_n_epochs=1,
#             enable_checkpointing=train_params["save_model"],
#             default_root_dir=model_checkpoint_subdir,
#             callbacks=[
#                 early_stopping, progress_bar]
#         )
        
#         # load datsets
#         train_dm = ASMGMLDataModule(
#             train_params["input_path"], train_params["batch_size"],
#             train_window_begin, train_window_end, period_val=val_period)

#         # train
#         val_trainer.fit(
#             model, datamodule=train_dm)
#         print(f"finished val_period {val_period}")

#         # log hyperparameters
#         val_trainer.logger.log_hyperparams({
#             **train_hparams,
#             'n_epochs': val_trainer.current_epoch - early_stopping.patience
#         }, metrics=early_stopping.best_score)

#         torch.manual_seed(train_params["seed"])
#         model.reset_parameters()


