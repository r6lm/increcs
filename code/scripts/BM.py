#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import numpy as np
import pandas as pd
import torch, torch.optim as optim, torch.nn as nn
import sys, os
from collections import defaultdict
sys.path.append("./..") # \todo: change for relative import
from dataset.ASMGMovieLens import ASMGMovieLens
from utils.save import get_timestamp, save_as_json, get_path_from_re

from torch.utils.data import DataLoader # is this tqdm importer?
from MF.model import get_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device = }')

# get model checkpoint path
timestamp = get_timestamp()


# # control flow parameters
# 

# In[ ]:


train_params = dict(
    input_path="../../data/preprocessed/ml_processed.csv",
    test_start_period=25,
    test_end_period=31, #26
    train_window=10,
    n_epochs=10,
    batch_size = 1024,
    learning_rate = 1e-3, # 1e-2 is the ASMG MF implementation
    model_checkpoint_dir = './../../model/MF/BM',
    model_filename_stem = 'first_mf',
    seed=6202,
    save_model=False,
    save_result=True
)
model_params = dict(
    alias="MF",
    n_users=43183,
    n_items=51149,
    n_latents=8
)


# In[ ]:


# initialize training components
torch.manual_seed(train_params["seed"])
model = get_model(model_params).to(device)
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=train_params["learning_rate"])


# In[ ]:



# initialize results container
res_dict = defaultdict(lambda: [])

# BU training regime routine
for test_period in range(
    train_params["test_start_period"], train_params["test_end_period"] + 1):

    # update periods
    train_window_begin = test_period - train_params["train_window"]
    train_window_end = test_period - 1 
    print(
        f"train periods: {train_window_begin}-{train_window_end}", 
        f"test period: {test_period}", sep="\n")

    # make checkpoint dir
    model_checkpoint_subdir = f'{train_params["model_checkpoint_dir"]}/'        f'{timestamp}' + (
            f'/T{test_period}' if train_params["save_model"] else "")
    if not os.path.exists(model_checkpoint_subdir):
        os.makedirs(model_checkpoint_subdir) 

    model_checkpoint_path = f'{model_checkpoint_subdir}/'         f'{train_params["model_filename_stem"]}.pth'

    # load datsets 
    train_data = ASMGMovieLens(
        train_params["input_path"], train_window_begin, train_window_end)

    for epoch in range(1, train_params["n_epochs"] + 1):

        cum_epoch_loss = 0.
        running_loss = 0.

        # set mopdel on training mode
        model.train()
        train_dataloader = DataLoader(
            train_data, batch_size=train_params["batch_size"], shuffle=True,
            num_workers=os.cpu_count())

        for i, (x_minibatch, y_minibatch) in enumerate(train_dataloader):

            # parse input
            user_minibatch = x_minibatch[:, 0].squeeze().to(device)
            item_minibatch = x_minibatch[:, 1].squeeze().to(device)

            # do not acumulate the gradients from last mini-batch
            optimizer.zero_grad()

            # get loss
            scores = model(user_minibatch, item_minibatch)
            loss = loss_function(scores, y_minibatch.to(device))
            ave_loss = loss.item()
            cum_epoch_loss += ave_loss

            # compute backpropagation gradients
            loss.backward()

            # update parameters
            optimizer.step()

            # report mean loss per item
            running_loss += ave_loss
            if i % 200 == 199:    # print every 200 mini-batches
                print(f'[{epoch = }, {i + 1:5d}]'
                    f'loss: {running_loss / 200:.4f}',
                    end="\r")
                running_loss = 0.

        # report epoch statistics
        epoch_loss = cum_epoch_loss * train_params["batch_size"] / len(train_data)
        print(f"\n{epoch_loss = :.4f}")

    else:
        res_dict["testPeriod"].append(test_period)
        res_dict["trainLoss"].append(epoch_loss)

    if train_params["save_model"]:

        # save model
        torch.save(model.state_dict(), model_checkpoint_path)
        print("saved model at:", model_checkpoint_path)

        # save json only once per training regime execution
        if test_period == train_params["test_start_period"]:
            save_as_json(
                {**train_params, **model_params}, 
                model_checkpoint_path.replace(".pth", "").replace(
                    f"/T{test_period}", ""))


    # load dataset
    test_data = ASMGMovieLens(train_params["input_path"], test_period)

    # test
    test_dataloader = DataLoader(
        test_data, batch_size=train_params["batch_size"], shuffle=False,
        num_workers=os.cpu_count())

    cum_test_loss = 0.
    running_loss = 0.


    model.to(device)

    with torch.no_grad():
        for i, (x_minibatch, y_minibatch) in enumerate(test_dataloader):

            # parse input
            user_minibatch = x_minibatch[:, 0].squeeze().to(device)
            item_minibatch = x_minibatch[:, 1].squeeze().to(device)


            # get loss
            scores = model(user_minibatch, item_minibatch)
            loss = loss_function(scores, y_minibatch.to(device))
            ave_loss = loss.item()
            cum_test_loss += ave_loss 

            # report mean loss per item
            running_loss += ave_loss
            if i % 1000 == 999:    # print every 1000 mini-batches
                print(f'running test loss: {running_loss / 1000:.4f}',
                    end="\r")
                running_loss = 0.
        else:

            # report epoch statistics
            test_loss = cum_test_loss * train_params["batch_size"] / len(test_data)
            print(f"\n{test_period = }\n{test_loss = :.4f}\n")
            res_dict["testLoss"].append(test_loss)
        
    # reset model parameters
    torch.manual_seed(train_params["seed"])
    model.reset_parameters()

else:
    df_path = model_checkpoint_path.replace(".pth", ".csv").replace(
        f"/T{test_period}", "")
    
    # calculate and display average loss
    res_df = pd.DataFrame(res_dict)
    average_srs = res_df.mean()
    average_srs.at["testPeriod"] = "mean"
    print(average_srs)
    if train_params["save_result"]: 
        pd.concat((res_df, average_srs.to_frame().T), axis=0, ignore_index=True
        ).to_csv(df_path, index=False)
        print(f"saved results csv at: {os.path.abspath(df_path)}")

