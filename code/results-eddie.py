#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from utils.save import load_json_array
# import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import glob, argparse
import sys
sys.path.append("./..") # \todo: change for relative import
from dataset.ASMGMovieLens import ASMGMovieLens
from collections import defaultdict
import psutil
import os


# In[ ]:


parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--regime", default="BM", help="training regime for whih results are collected")
parser.add_argument(
    "--eddie", default="1", choices=[f"{i}" for i in range(2)],
    help="training regime for whih results are collected")

# parsed_args = parser.parse_args(["--eddie", "0"])
parsed_args = parser.parse_args()

run_on_eddie = bool(int(parsed_args.eddie))

params_dict = dict(
    lightning_seeds=range(1, 6),
    regimes=("BM", "IU", "BIU"),
    input_path = "../data/preprocessed/ml_processed.csv",
    tyxe_dir="/home/s2110626/diss/TyXe",
    tyxe_versions=range(73, 78),
    start_test_period = 25,
    end_test_period = 31
)
if not run_on_eddie:
    params_dict.update(dict(
        lightning_seeds = [6202],
        tyxe_versions = [60],
        tyxe_dir="../../others/TyXe"
    ))

params = argparse.Namespace(**params_dict)
params


# # Get Generalization Error 

# In[ ]:


# for seed in range(1,6):
def mean_se_log(concat_prob_correct):
    prob_correct_clean = concat_prob_correct[concat_prob_correct != 0]
    print(f"predictions completely wrong: {len(concat_prob_correct) - len(prob_correct_clean)}")
    log_prob_correct = torch.log(prob_correct_clean)
    std_log, mean_log = torch.std_mean(log_prob_correct)
    se_log = std_log / np.sqrt(len(prob_correct_clean))
    return mean_log,se_log

summary_dict = {}
preds_dict = defaultdict(lambda: [])
pred_labels_dict = defaultdict(lambda: [])

for regime in params.regimes:
    print(f"{regime =}")

    # initialize containers 
    mc_means = []
    mc_se = []

    # get y_true and global parameters
    concat_true_y = ASMGMovieLens(
        params.input_path, params.start_test_period, 
        params.end_test_period).y
    n_periods = params.end_test_period - params.start_test_period + 1
    obs_per_period = int(len(concat_true_y) / n_periods)

    if regime != "BIU":
        iterate_in = params.lightning_seeds

    else: 
        iterate_in = params.tyxe_versions

    for seed_or_version in iterate_in:

        print(f"{seed_or_version = }")

        # concatenate results
        concat_prob_correct = torch.ones_like(concat_true_y) * -1
        concat_pred_label_y = torch.ones_like(concat_true_y, dtype=bool)
        for i, test_period in enumerate(range(
            params.start_test_period, params.end_test_period + 1)):

            # get preds filename
            if regime != "BIU":
                filename = (
                    f"../model/MF/{regime}/T{test_period}/preds-s"
                    f"{seed_or_version}.pt")
            else:
                filenm_pattern = (
                    f"{params.tyxe_dir}/model/MF/mean-field/version_"
                    f"{seed_or_version}/T{test_period}/preds-s*.pt")
                matching_files = glob.glob(filenm_pattern)
                assert len(matching_files) == 1, (
                    "More than 1 file matches the pattern")
                filename = matching_files[0]

            pred_y = torch.load(filename)

            # obtain accuracy
            pred_label_y = torch.round(pred_y)
            concat_pred_label_y[
                i * obs_per_period:(i + 1) * obs_per_period
            ] = pred_label_y

            # obtain probability of correct prediction
            true_y = concat_true_y[i * obs_per_period:(i + 1) * obs_per_period]
            prob_correct = pred_y
            prob_correct[~true_y.bool()] = 1 - prob_correct[~true_y.bool()]
            concat_prob_correct[
                i * obs_per_period:(i + 1) * obs_per_period
            ] = prob_correct

        assert (concat_prob_correct != -1).all().item()

        mean_log, se_log = mean_se_log(concat_prob_correct)
        mc_means.append(- mean_log.item())
        mc_se.append(se_log.item())
        preds_dict[regime].append(concat_prob_correct)
        pred_labels_dict[regime].append(concat_pred_label_y)

    summary_dict.update({
        (regime, "mean"): mc_means,
        (regime, "se"): mc_se
    })

    gen_error = np.mean(mc_means)
    gen_error_se = np.mean(mc_se)
    print(f"generalization error: {gen_error:.4f} ± {gen_error_se:.4f}\n\n")


# In[ ]:


print(pd.DataFrame(summary_dict).to_csv())


# In[ ]:


from scipy.stats import wilcoxon, ttest_rel
from itertools import combinations


# In[ ]:


print("loss")
cat_loss_dict = {k: - torch.log(torch.cat(v)) for k, v in preds_dict.items()}
for k1, k2 in combinations(cat_loss_dict, 2):
    print("\n", k1, k2)
    print("greater")
    print(wilcoxon(cat_loss_dict[k1], cat_loss_dict[k2], alternative="greater"))
    print("less")
    print(wilcoxon(cat_loss_dict[k1], cat_loss_dict[k2], alternative="less"))


# In[ ]:


for k1, k2 in combinations(cat_loss_dict, 2):
    print("\n", k1, k2)
    print("greater")
    print(ttest_rel(cat_loss_dict[k1], cat_loss_dict[k2], alternative="greater"))
    print("less")
    print(ttest_rel(cat_loss_dict[k1], cat_loss_dict[k2], alternative="less"))


# In[ ]:


print("accuracy")
cat_labels_dict = {
    k: torch.cat([ypred == concat_true_y.bool() for ypred in v]).int(
    ) for k, v in pred_labels_dict.items()}
for k1, k2 in combinations(cat_labels_dict, 2):
    print("\n", k1, k2)
    print("greater")
    print(wilcoxon(cat_labels_dict[k1], cat_labels_dict[k2], alternative="greater"))
    print("less")
    print(wilcoxon(cat_labels_dict[k1], cat_labels_dict[k2], alternative="less"))


# In[ ]:


pid = os.getpid()
python_process = psutil.Process(pid)
memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
print('memory use:', memoryUse)

