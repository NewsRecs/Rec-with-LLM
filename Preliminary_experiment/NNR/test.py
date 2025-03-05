# %%
from config import model_name
import pandas as pd
from tqdm import tqdm
import os
import random
import numpy as np
import csv
import importlib
import copy
from datetime import timedelta
try:
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()

import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

def str_to_timestamp(string):
    return datetime.timestamp(datetime.strptime(string,'%Y-%m-%d %H:%M:%S'))
model_name = 'NRMS'
ratio = 8
lifetime = 36
preprocess_data_folder = './data/preprocessed_data'
preprocess_type = '(type1)'         # '(type1)'/ '(type2)' / ''
data_list = ['Adressa_7w'
# ,'Adressa_5w','Adressa_6w'
]
data_list = [s+preprocess_type for s in data_list]

for data_type in data_list:
    print(f'\n{data_type} / {model_name} \nmake_behaviors_parsed')
    news_dir = f'./data/preprocessed_data/{data_type}'
    train_dir = f'./data/preprocessed_data/{data_type}/train'
    # behaviors = pd.read_table(
    #     os.path.join(train_dir,f'behaviors_parsed_ns{ratio}_lt{lifetime}.tsv'))
    # behaviors['click'] = behaviors['candidate_news_impre'].str.split(' ')[0]
    
    behaviors = pd.read_table(
        os.path.join(train_dir,'behaviors.tsv'),
        header=None,
        names=['user', 'time', 'history', 'click'])
    behaviors.click = behaviors.click.str.split('-').str[0]
    behaviors.history.fillna(' ', inplace=True)

    # %%
    #### edited JW
    behaviors['time'] = pd.to_datetime(behaviors['time'])
    ##
    added_columns_list = []

    news = pd.read_table(os.path.join(news_dir, 'total_news(raw).tsv'),
                        quoting=csv.QUOTE_NONE,
                        header=None,
                        names=['id','category', 'subcategory','title','body','raw_id','publish_time','clicks'])  # TODO try to avoid csv.QUOTE_NONE

    news['publish_time'] = pd.to_datetime(news['publish_time'])
    news = news.set_index(['id'])
    news.fillna(' ', inplace=True)
    news['train_click'] = 0

    train_click_cnt = behaviors['click'].value_counts()
    for i,row in tqdm(news.iterrows(),desc="News training click count"):
        try:
            news.at[i,'train_click'] = train_click_cnt[i]
        except:
            pass

    news = news.sort_values(by=['train_click'], ascending=False)

    user_poslist = behaviors.groupby(by='user')['click'].apply(lambda x: ','.join(x))
# %%
    dead_cnt = 0
    for i,row in tqdm(behaviors.iterrows()):
        if news.loc[row.click].publish_time + np.timedelta64(36,'h')  > row.time:
            dead_cnt += 1
    print('죽은 클릭 수:',dead_cnt)
    print('살아있는 클릭 수:',len(behaviors)-dead_cnt)
# %%

# %%
510611/(510611 + 3879709)*100
# %%
end_time = behaviors.iloc[0].time - np.timedelta64(36,'h') - timedelta(days=3*7)
mask = (news['publish_time'] >= end_time)
news_negative_candidate = news.loc[mask]
# %%
len(news)
# %%
len(news_negative_candidate)
# %%
3059/25353
# %%
