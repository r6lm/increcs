#!/usr/bin/env python
# coding: utf-8

# # Notes
# 
# The number of movies observed in the "ratings.csv" (59,047) does not match the number of movies on the readme (62,423)

# In[1]:


import pandas as pd
import os

get_ipython().run_line_magic('matplotlib', 'inline')
# sns.set(font_scale = 1.5)


# In[2]:


# parameters
preproc_config = dict(
    data_dir = "./../../data",
    n_periods = 31,
    dst_dir = "preprocessed",
    dst_filename = "ml_processed",
    movie_map_filename = "movieId_map"
    )
print(preproc_config)


# In[3]:


# raw data
parsed_ratings_df = pd.read_csv(
    f'{preproc_config["data_dir"]}/ml-25m/ratings.csv') #, 
    #dtype=dict(zip(cat_cols, ["category"] * 2)))


# parsed_ratings_df.info()

# parsed_ratings_df[cat_cols].nunique()

# tt["userId"]

# tt.nunique()

# In[4]:


# downcast ratings to uints from 1 to 10
parsed_ratings_df.loc[:, "rating"] = (parsed_ratings_df.rating * 2).astype("uint8")

# parse timestamps as datetimes
parsed_ratings_df.loc[:, "timestamp"] = pd.to_datetime(parsed_ratings_df["timestamp"], unit="s")

# print result of parsing
# parsed_ratings_df.info()


# In[5]:


# get ASMG subset data
start_date = "20140101"
end_date = "20190101"
asmg_ratings_df = parsed_ratings_df.loc[lambda df: df["timestamp"].between(
    start_date, end_date, inclusive="left"), :].copy()



# set user according to ASMG: give the id according to their order
def get_id_map(srs): return pd.Series(
    srs.unique()).reset_index(drop=False).set_index(0).squeeze()


user_map_srs = get_id_map(asmg_ratings_df["userId"])
asmg_ratings_df.loc[:, "userId"] = asmg_ratings_df.loc[:, "userId"].map(
    user_map_srs).astype("category")

# drop unused categories on movieId
item_map_srs = get_id_map(asmg_ratings_df["movieId"])
asmg_ratings_df.loc[:, "movieId"] = asmg_ratings_df.loc[:, "movieId"].map(
    item_map_srs).astype("category")

# assign label 1 to ratings >= 8 to transform in binary classification
asmg_ratings_df = asmg_ratings_df.assign(**{
    "label": lambda df: df["rating"].ge(8).astype("uint8")
})

# drop ratings as it is unused
asmg_ratings_df.drop("rating", axis=1, inplace=True)


# In[6]:


# note this statistics do not coincide with paper but do coincide with code 
# output of proproc/ml_preproc.py
print("dataset statistics:", asmg_ratings_df.nunique(), "", sep="\n")


# asmg_ratings_df.info()

# In[7]:


# build period variable 
asmg_ratings_df.sort_values("timestamp", ascending=True, inplace=True)
asmg_ratings_df.reset_index(inplace=True, drop=True)
total_obs_before_drop = len(asmg_ratings_df)
records_per_period, lost_records = divmod(
    total_obs_before_drop, preproc_config['n_periods'])
records_per_period, lost_records
print(f"""{records_per_period = }
{lost_records = }""")
asmg_ratings_df = asmg_ratings_df.assign(**{
    "period": lambda df: (df.index // records_per_period + 1).astype("uint8")
})
asmg_ratings_df.drop(asmg_ratings_df.index[
    asmg_ratings_df["period"] > preproc_config['n_periods']], inplace=True)

# drop timestamp as it is unused
asmg_ratings_df.drop("timestamp", axis=1, inplace=True)

# check types and disk usage
asmg_ratings_df.info()


# In[8]:


# period_df = asmg_ratings_df.groupby('period')['timestamp'].agg(['count', 'min', 'max'])


# In[9]:


# make directory
preprocessed_data_dir =     f'{preproc_config["data_dir"]}/{preproc_config["dst_dir"]}'
if not os.path.exists(preprocessed_data_dir):
    os.mkdir(preprocessed_data_dir)

# save processed file which is the input of all models
dst_path = f"{preprocessed_data_dir}/{preproc_config['dst_filename']}.csv"
asmg_ratings_df.to_csv(dst_path, index=False)
print(f"saved output at {os.path.abspath(dst_path)}")


# In[14]:


item_map_srs.index.to_series().rename("movieId")


# In[15]:


# save movieId map for movie reidentification
movie_map_path = f'{preprocessed_data_dir}/{preproc_config["movie_map_filename"]}.csv'
item_map_srs.index.to_series().rename("movieId").reset_index(
    drop=True).to_csv(movie_map_path)
print(f"saved movieId_map at {os.path.abspath(movie_map_path)}")

