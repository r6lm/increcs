#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse

# parameters to tune on Eddie
parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed", default="6202", help="random seed for reproducibility")

# parsed_args = parser.parse_args([])
parsed_args = parser.parse_args()
parsed_args

# fast access parameters
validation_mode = False
fast_dev_run = False
run_on_eddie = True


# In[ ]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import numpy as np
import pandas as pd
import torch, torch.optim as optim, torch.nn as nn
import sys, os
from collections import defaultdict
from time import time
from sklearn.metrics import roc_auc_score
sys.path.append("./..") # \todo: change for relative import
from dataset.ASMGMovieLens import ASMGMLDataModule
from utils.save import (get_timestamp, save_as_json, get_path_from_re, 
    append_json_array, load_json_array, get_version)
from utils.performance import auc

from torch.utils.data import DataLoader
from MF.model import get_model, MF
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# filter warning for not setting validation set
import warnings
warnings.filterwarnings(
    "ignore", "Total length of `DataLoader` across ranks is zero.*")

start_time = time()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device = }')

# get training regime experiment id
# timestamp = get_timestamp()


# # control flow parameters
# 

# In[ ]:


train_params = dict(
    input_path="../../data/preprocessed/ml_processed.csv",
    val_start_period=11,
    val_end_period=24,
    test_start_period=25, # change to None if running as validation
    test_end_period=31,  # 25
    train_window=10,
    seed=6202,
    model_filename='first_mf',
    base_path=None,  # "../../model/MF/IU/base/220707T162306/first_mf.ckpt",
    save_model=False,
    save_result=True
)
# these are stored on the tensorboard logs
model_params = dict(
    alias="MF",
    n_users=43183,
    n_items=51149,
    n_latents=8,
    l2_regularization_constant=1e-6,
    # moved
    learning_rate=1e-3,  # 1e-2 is the ASMG MF implementation
    batch_size=1024,
    n_epochs_offline=20,
    n_epochs_online=5,
    early_stopping_online=False # train_params["test_start_period"] is None,
)
train_params["model_checkpoint_dir"] = f'./../../model/{model_params["alias"]}/IU'

# in this validation mode no test period is run because we are trying to
# infer the number of epochs on the online period
if validation_mode:
    train_params.update(dict(
        test_start_period=None,
        test_end_period=None,
    ))
    model_params.update(dict(
        n_epochs_online=30,
    ))


# saved as json to be able to replicate the experiment
experiment_params = {**model_params, **train_params}

# ensure fast dev and early stopping are not both true because the logged losses 
# do not exist
assert not (fast_dev_run and model_params["early_stopping_online"])


# In[ ]:


# initialize training components
torch.manual_seed(train_params["seed"])
model = get_model(model_params)#.to(device)


# progess log
progress_bar = TQDMProgressBar(refresh_rate=200)


# # train base

# In[ ]:


if train_params["base_path"] is None:
    
    # IU base training

    # get model checkpoint path
    timestamp = get_timestamp()

    # update periods
    train_window_begin = train_params["val_start_period"] - train_params["train_window"]
    train_window_end = train_params["val_start_period"] - 1
    print(
        f"base train periods: {train_window_begin}-{train_window_end}")

    # make checkpoint dir
    model_checkpoint_subdir = f'{train_params["model_checkpoint_dir"]}' + (
        f'/base/{timestamp}')
    if not os.path.exists(model_checkpoint_subdir):
        os.makedirs(model_checkpoint_subdir)

    base_trainer = Trainer(
        accelerator="auto", 
        devices=1 if (torch.cuda.is_available() or fast_dev_run) else 0,
        max_epochs=model_params["n_epochs_offline"],
        reload_dataloaders_every_n_epochs=1, enable_checkpointing=False,
        default_root_dir=model_checkpoint_subdir, logger=False, callbacks=[
            progress_bar], fast_dev_run=fast_dev_run, deterministic=True
    )
    
    # load datsets
    train_dm = ASMGMLDataModule(
        train_params["input_path"], model_params["batch_size"],
        train_window_begin, train_window_end, run_on_eddie=run_on_eddie)

    # train
    base_trainer.fit(
        model, datamodule=train_dm)
    print(f"finished base training")

    # save 
    train_params["base_path"] =         f"{model_checkpoint_subdir}/{train_params['model_filename']}.ckpt"
    base_trainer.save_checkpoint(train_params["base_path"])
    save_as_json(
        {**model_params, **train_params}, 
        train_params["base_path"].replace(".ckpt", ""))

else:
    print("loading base model at:", train_params["base_path"])
    model = get_model(
        experiment_params, return_instance=False).load_from_checkpoint(
            checkpoint_path=train_params["base_path"])

# ensure reproducibility
torch.manual_seed(train_params["seed"])


# # transfer

# In[ ]:


# `test_start_period` is used as a lever for use the transfer cycle for
# searching hyperparameters
if train_params["test_start_period"] is None:

    print("running online cycle until end of validation")


train_hparams = {
    'learning_rate': model_params["learning_rate"],
    'l2_regularization_constant': model_params["l2_regularization_constant"]
}
val_list = []
version = None
transfer_callbacks = [progress_bar]

# IU validation routine
for val_period in range(
    train_params["val_start_period"] + 1,
    train_params["val_end_period"] + 1):

    # update periods
    train_period = val_period - 1
    print(
        f"train period: {train_period}",
        f"validaton period: {val_period}", sep="\n")

    # make checkpoint dir
    model_checkpoint_subdir = f'{train_params["model_checkpoint_dir"]}/' + (
        f'/transfer')
    if not os.path.exists(model_checkpoint_subdir):
        os.makedirs(model_checkpoint_subdir)

    # set logger
    version = get_version(
        model_checkpoint_subdir) if version is None else version
    
    if val_period == train_params["val_start_period"] + 1:
        print(f"experiment version: {version}")
    
    logger = TensorBoardLogger(model_checkpoint_subdir, version=version,
                               sub_dir=f"V{val_period:02}")

    # save experimet json so it can be replicated
    if train_params["save_model"] or train_params["save_result"]:
        
        # save experimet json only once per training regime execution
        if val_period == train_params["val_start_period"] + 1:
            save_as_json(
                experiment_params, 
                f'{model_checkpoint_subdir}/{version}'
                )

    # use early stopping if the script is running for selecting hyperparameters
    if model_params["early_stopping_online"]:
        early_stopping = EarlyStopping(
            monitor="val_loss", mode="min", verbose=True, min_delta=1e-4)
        transfer_callbacks.append(early_stopping)

    val_trainer = Trainer(
        accelerator="auto", 
        devices=1 if (torch.cuda.is_available() or fast_dev_run) else 0,
        max_epochs=model_params["n_epochs_online"],
        reload_dataloaders_every_n_epochs=1,
        enable_checkpointing=train_params["save_model"],
        default_root_dir=model_checkpoint_subdir, logger=logger,
        callbacks=transfer_callbacks, enable_model_summary=False,
        fast_dev_run=fast_dev_run, deterministic=True)

    # load datsets
    train_dm = ASMGMLDataModule(
        train_params["input_path"], model_params["batch_size"],
        train_period, period_val=val_period, run_on_eddie=run_on_eddie)

    # train
    val_trainer.fit(
        model, datamodule=train_dm)
    print(f"finished val_period {val_period}")

    # log hyperparameters according to transfer period mode
    val_score = (
        early_stopping.best_score if model_params[
            "early_stopping_online"] else val_trainer.logged_metrics[
                "val_loss"]).item()
    n_epochs_dict = {}
    if model_params["early_stopping_online"]:
        n_epochs_dict["n_epochs_online"] = val_trainer.current_epoch - (
            early_stopping.patience)
    val_trainer.logger.log_hyperparams(
        {
            **model_params,
            **n_epochs_dict
        },
        metrics=val_score)

    val_list.append({
        "period": val_period,
        **model_params,
        "val_loss": val_score})

else:
    # save validation results
    val_df = pd.DataFrame(val_list)
    average_srs = val_df.mean()
    average_srs.at["period"] = "mean"
    print(average_srs)
    val_df_path = f'{model_checkpoint_subdir}/{version}.csv'
    if train_params["save_result"]:
        pd.concat((val_df, average_srs.to_frame().T), axis=0, ignore_index=True
                  ).to_csv(val_df_path, index=False)
        print(f"saved results csv at: {os.path.abspath(val_df_path)}")


# # test 

# In[ ]:


if train_params["test_start_period"] is not None:

    # initialize results container
    res_dict = defaultdict(lambda: [])

    print("running online cycles until end of test")
    
    # BU test routine
    for test_period in range(
        train_params["test_start_period"], train_params["test_end_period"] + 1):

        # update periods
        train_period = test_period - 1 
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
            accelerator="auto", 
            devices=1 if (torch.cuda.is_available() or fast_dev_run) else 0,
            max_epochs=model_params["n_epochs_online"], reload_dataloaders_every_n_epochs=1,
            enable_checkpointing=train_params["save_model"],
            default_root_dir=model_checkpoint_subdir,
            logger=False, enable_model_summary=False,
            callbacks=[progress_bar], fast_dev_run = fast_dev_run, 
            deterministic=True
        )

        # load datsets 
        train_dm = ASMGMLDataModule(
            train_params["input_path"], model_params["batch_size"], 
            train_period, run_on_eddie=run_on_eddie)
        test_dm = ASMGMLDataModule(
            train_params["input_path"], model_params["batch_size"], test_period,
            run_on_eddie=run_on_eddie)

        # train
        start_training_time = time()
        test_trainer.fit(
            model, datamodule=train_dm)
        res_dict["train_time"].append(time() - start_training_time)
        res_dict["period"].append(test_period)
        res_dict["trainLoss"].append(test_trainer.logged_metrics["train_loss"].item())
        print(f"finished test_period {test_period}")

        # get test loss
        test_trainer.test(model, datamodule=test_dm)
        res_dict["loss"].append(test_trainer.logged_metrics["test_loss"].item())

        # get test AUC
        with torch.no_grad():
            test_auc = auc(model, test_dm.test_dataset.y, test_dm.test_dataloader(), 
                model_params["batch_size"])
        res_dict["auc"].append(test_auc)

        
    else:
        timestamp = get_timestamp()
        df_path = f"{model_checkpoint_subdir.replace(f'/T{test_period:02}', '')}/{version}.csv"
        df_path

        # calculate and display average loss
        res_df = pd.DataFrame(res_dict)

        average_srs = res_df.mean()
        average_srs.at["period"] = "mean"
        print(average_srs)

        if train_params["save_result"]: 
            pd.concat((res_df, average_srs.to_frame().T), axis=0, ignore_index=True
            ).to_csv(df_path, index=False)
            print(f"saved results csv at: {os.path.abspath(df_path)}")

else:
    print("skipped test cycle")


# In[ ]:


# define where to save master results
results_master_path = train_params["model_checkpoint_dir"] + "/results.json"

# concatenate summary results and script args
res_dict = average_srs.to_dict()
res_dict.update(**vars(parsed_args), **{
    "n_epochs_on_test": model_params["n_epochs_offline"],
    "timestamp": timestamp
    })
print(res_dict)
append_json_array(res_dict, results_master_path)
load_json_array(results_master_path)


# In[ ]:


print(f"Total elapsed time: {(time() - start_time) / 3600:.2f} hrs.")


# In[ ]:




