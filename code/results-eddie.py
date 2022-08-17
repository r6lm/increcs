#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utils.save import load_json_array
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import glob, argparse

sys.path.append("./..") # \todo: change for relative import
from dataset.ASMGMovieLens import ASMGMovieLens


# # Get Generalization Error 

# In[2]:


parser = argparse.ArgumentParser()
parser.add_argument(
    "--regime", default="BM", help="training regime for whih results are collected")

# parsed_args = parser.parse_args([])
parsed_args = parser.parse_args()
print(parsed_args.regime)


# In[3]:


mc_means = []
mc_se = []

for seed in range(1,6):
# for seed in ["6202"]:

    print(f"{seed = }")

    # concatenate results
    preds_pattern = f"../model/MF/BM/T*/preds-s{seed}.pt"
    start_test_period, end_test_period = (25, 31)
    n_periods = end_test_period - start_test_period + 1
    concat_true_y = ASMGMovieLens("../data/preprocessed/ml_processed.csv", 25, 31).y
    obs_per_period = int(len(concat_true_y) / n_periods)
    concat_prob_correct = torch.ones_like(concat_true_y) * -1
    for i, filename in enumerate(sorted(glob.glob(preds_pattern))):
        pred_y = torch.load(filename)
        true_y = concat_true_y[i * obs_per_period:(i + 1) * obs_per_period]
        prob_correct = pred_y
        prob_correct[~true_y.bool()] = 1 - prob_correct[~true_y.bool()]
        concat_prob_correct[
            i * obs_per_period:(i + 1) * obs_per_period
        ] = prob_correct

    assert (concat_prob_correct != -1).all().item()


    prob_correct_clean = concat_prob_correct[concat_prob_correct != 0]
    print(f"predictions completely wrong: {len(concat_prob_correct) - len(prob_correct_clean)}")
    log_prob_correct = torch.log(prob_correct_clean)
    std_log, mean_log = torch.std_mean(log_prob_correct)
    se_log = std_log / np.sqrt(len(prob_correct_clean))
    mc_means.append(mean_log.item())
    mc_se.append(se_log.item())

gen_error = np.mean(mc_means)
gen_error_se = np.mean(mc_se)
print(f"generalization error: {gen_error:.4f} ± {gen_error_se:.4f}\n\n")

