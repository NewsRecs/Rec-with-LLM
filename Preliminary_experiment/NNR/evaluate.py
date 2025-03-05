import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import torch
from config import model_name
from torch.utils.data import Dataset, DataLoader
from os import path
import sys
import pandas as pd
from ast import literal_eval
import importlib
from datetime import timedelta, datetime
import csv


# 1. 모델 및 설정 불러오기
try:
    Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()

# 2. 장치 설정 (GPU 사용 가능 시 GPU 사용)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 3. 성능 지표 계산 함수
def sigmoid(x,a=1):
    if x < -50/a:
        return 0
    elif x > 50/a:
        return 1
    return 1 / (1 +np.exp(-a*x))

def dcg_score(y_true, y_score, k=10):
    """ DCG 점수 계산 (상위 k개) """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    """ nDCG 점수 계산 """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true, y_score):
    """ MRR 점수 계산 """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def value2rank(d):
    values = list(d.values())
    ranks = [sorted(values, reverse=True).index(x) for x in values]
    return {k: ranks[i] + 1 for i, k in enumerate(d.keys())}

def accurate_score(y_true, y_pred):
    """
    Accurate 점수 계산: positive를 1위로 맞췄는지 평가
    Args:
        y_true: 실제 레이블 (list 또는 numpy array)
        y_pred: 모델의 점수 (list 또는 numpy array)
    Returns:
        accurate: 1 (positive가 1위인 경우) 또는 0 (그 외)
    """
    # 점수를 기준으로 정렬
    order = np.argsort(y_pred)[::-1]  # 내림차순 정렬
    # 1위의 실제 레이블 확인
    top_label = y_true[order[0]]
    # 1위가 positive(1)인지 확인
    return 1 if top_label == 1 else 0


# 4. 데이터셋 클래스 정의
class NewsDataset(Dataset):
    """
    뉴스 데이터를 로드하고 전처리하는 클래스
    """
    def __init__(self, news_path, roberta_embedding_dir):
        super(NewsDataset, self).__init__()
        self.news_parsed = pd.read_table(
            news_path,
            usecols=['id'] + config.dataset_attributes['news'],
            converters={
                attribute: literal_eval
                for attribute in set(config.dataset_attributes['news']) & set([
                    'title', 'abstract', 'title_entities', 'abstract_entities',"category_word",
                    'title_roberta', 'title_mask_roberta', 'abstract_roberta',
                    'abstract_mask_roberta'
                ])
            })
        self.news2dict = self.news_parsed.to_dict('index')
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                if type(self.news2dict[key1][key2]) != str:
                    self.news2dict[key1][key2] = torch.tensor(
                        self.news2dict[key1][key2])

    def __len__(self):
        return len(self.news_parsed)

    def __getitem__(self, idx):
        item = self.news2dict[idx]
        if model_name == 'Exp2' and not config.fine_tune:
            for k in set(config.dataset_attributes['news']) & set(
                ['title', 'abstract']):
                item[k] = self.roberta_embedding[k][idx]
        return item


class UserDataset(Dataset):
    """
    Load users for evaluation, duplicated rows will be dropped
    사용자 데이터를 로드하고 전처리하는 클래스
    """
    def __init__(self, behaviors_path, user2int_path):
        super(UserDataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                    #    usecols=[1, 3],
                                       usecols=[1, 2],
                                       names=['user', 'clicked_news'])
        # self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors['clicked_news'] = self.behaviors['clicked_news'].fillna(' ')
        self.behaviors.drop_duplicates(inplace=True)
        user2int = dict(pd.read_table(user2int_path).values.tolist())
        user_total = 0
        user_missed = 0
        for row in self.behaviors.itertuples():
            user_total += 1
            if row.user in user2int:
                self.behaviors.at[row.Index, 'user'] = user2int[row.user]
            else:
                user_missed += 1
                self.behaviors.at[row.Index, 'user'] = 0
        if model_name == 'LSTUR':
            print(f'User miss rate: {user_missed/user_total:.4f}')

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "user":
            row.user,
            "clicked_news_string":
            row.clicked_news,
            "clicked_news":
            row.clicked_news.split()[:config.num_clicked_news_a_user]
        }
        item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = config.num_clicked_news_a_user - len(
            item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = ['PADDED_NEWS'
                                ] * repeated_times + item["clicked_news"]

        return item


class BehaviorsDataset(Dataset):
    """
    Load behaviors for evaluation, (user, time) pair as session
    행동 데이터를 로드하고 전처리하는 클래스
    """
    def __init__(self, behaviors_path):
        super(BehaviorsDataset, self).__init__()
        self.behaviors = pd.read_table(behaviors_path,
                                       header=None,
                                       usecols=range(4),
                                       names=[
                                           'impression_id', 'user',
                                           'clicked_news', 'impressions'
                                       ]
                                    #    usecols=range(5),
                                    #    names=[
                                    #        'impression_id', 'user', 'time',
                                    #        'clicked_news', 'impressions'
                                    #    ]
                                       )
        # self.behaviors.clicked_news.fillna(' ', inplace=True)
        self.behaviors['clicked_news'] = self.behaviors['clicked_news'].fillna(' ')
        self.behaviors.impressions = self.behaviors.impressions.str.split()

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]
        item = {
            "impression_id": row.impression_id,
            "user": row.user,
            # "time": row.time,
            "clicked_news_string": row.clicked_news,
            "impressions": row.impressions
        }
        return item


@torch.no_grad()
def evaluate(model, directory, max_count=sys.maxsize):
    """
    모델 평가 함수
    Args:
        model: model to be evaluated
        directory: the directory that contains two files (behaviors.tsv, news_parsed.tsv)
    Returns:
        AUC
        nMRR
        nDCG@5
        nDCG@10
    """
    news_dataset = NewsDataset(path.join(directory, 'news_parsed.tsv'),
                               path.join(directory, 'roberta'))
    news_dataloader = DataLoader(news_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 drop_last=False,
                                 pin_memory=True)

    user_dataset = UserDataset(path.join(f'{config.data_folder}/test', config.test_behaviors_file),
                               f'{config.data_folder}/user2int.tsv')
    user_dataloader = DataLoader(user_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=config.num_workers,
                                 drop_last=False,
                                 pin_memory=True)

    behaviors_dataset = BehaviorsDataset(path.join(f'{config.data_folder}/test', config.test_behaviors_file))
    behaviors_dataloader = DataLoader(behaviors_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=config.num_workers)

    aucs = []
    mrrs = []
    ndcg5s = []
    ndcg10s = []
    accs = []

    count = 0
    news_df = pd.read_table(path.join(f'{config.data_folder}', 'total_news(raw).tsv'),
                        quoting=csv.QUOTE_NONE,
                        header=None,
                        names=['id','category', 'subcategory','title','body','raw_id','publish_time','clicks'])  # TODO try to avoid csv.QUOTE_NONE

    news_df['publish_time'] = pd.to_datetime(news_df['publish_time'])
    news_df = news_df.set_index(['id'])
    # news_df.fillna(' ', inplace=True)
    news_df = news_df.fillna(' ')

    news2vector = {}
    for minibatch in tqdm(news_dataloader,
                          desc="Calculating vectors for news"):
        news_ids = minibatch["id"]
        
        if any(id not in news2vector for id in news_ids):
            if model_name.startswith('FIM'):
                news_vector_d0, news_vector_dL = model.get_news_vector(minibatch)
                # print(news_vector_d0.shape)
                # print(news_vector_dL.shape)
                
                for id in news_ids:
                    if id not in news2vector:
                        vector = [news_vector_d0, news_vector_dL]
                        news2vector[id] = vector
                    # print(vector)
                    # print(vector[0].shape)
                    # print(vector[1].shape)
            else:
                news_vector = model.get_news_vector(minibatch)
                for id, vector in zip(news_ids, news_vector):
                    if id not in news2vector:
                        news2vector[id] = vector
    if model_name.startswith('FIM'):
        news2vector['PADDED_NEWS'] = [torch.zeros(
            list(news2vector.values())[0][0].size()), torch.zeros(
            list(news2vector.values())[0][1].size())]
    else:
        news2vector['PADDED_NEWS'] = torch.zeros(
            list(news2vector.values())[0].size())  # news2vector.values())[0].size() = torch.Size [100]
    # #####    
    # for minibatch in tqdm(behaviors_dataloader,
    #                       desc="Calculating probabilities"):
    #     print(minibatch['clicked_news_string'][0])
    #     print()
    #     print()
    #     break
    # for minibatch in tqdm(user_dataloader,
    #                       desc="Calculating vectors for users"):
    #     user_strings = minibatch["clicked_news_string"]
    #     print(minibatch["clicked_news_string"])
    #     break
    # #####
    # assert a == 1

    # 사용 x
    # if model_name.startswith('FIM'):
    #     for minibatch in tqdm(behaviors_dataloader, desc="Calculating probabilities"):
    #         count += 1
    #         if count == max_count:
    #             break
    #         impre_news_list = [news[0].split('-')[0] for news in minibatch['impressions']]
    #         if config.test_filter != False:
    #             minibatch['time'] = pd.to_datetime(minibatch['time'])
    #             end_time = minibatch['time'].values[0] - np.timedelta64(config.lifetime,'h')
    #             if news_df.loc[impre_news_list[0]]['publish_time'] < end_time:
    #                 continue
    #         history_news_list = minibatch["clicked_news_string"][0].split()
    #         padding_num = 50 - len(history_news_list)
    #         if padding_num > 0:
    #             minibatch["clicked_news_string"][0] = 'PADDED_NEWS '* padding_num + minibatch["clicked_news_string"][0]
    #             history_news_list = minibatch["clicked_news_string"][0].split()
    #         elif padding_num <0:
    #             history_news_list = history_news_list[-50:]
                
    #         clicked_news_vector_d0 = torch.stack([news2vector[x][0].to(device) 
    #                         for x in history_news_list],
    #                         dim=1).squeeze(dim=2)
    #         clicked_news_vector_dL = torch.stack([news2vector[x][1].to(device) 
    #                         for x in history_news_list],
    #                         dim=1).squeeze(dim=2)
    #         # print('clicked_news_vector:',clicked_news_vector_d0.shape)
    #         # print('test_clicked_news_vector_dL:',clicked_news_vector_dL.shape)
            
    #         candidate_news_vector_d0 = torch.stack([news2vector[news][0].to(device) for news in impre_news_list],dim=1).squeeze(dim=2)
    #         candidate_news_vector_dL = torch.stack([news2vector[news][1].to(device) for news in impre_news_list],dim=1).squeeze(dim=2)
    #         # print('candidate_news_vector_d0',candidate_news_vector_d0.shape)
    #         # print('candidate_news_vector_dL',candidate_news_vector_dL.shape)
    #         click_probability = model.get_prediction(candidate_news_vector_d0, candidate_news_vector_dL,
    #                                                 clicked_news_vector_d0, clicked_news_vector_dL).squeeze()
    #         y_pred = click_probability.tolist()

    #         # 사용 X (수명고려 안함함)
    #         # if config.test_filter != False:
    #         #     news_publish_time = news_df.loc[impre_news_list]['publish_time'].to_list()
    #         #     news_age = [(minibatch['time'].values[0]-publish_time).total_seconds()/3600.00 for publish_time in news_publish_time ]
    #         #     freshness_filter = [sigmoid(36-age,config.test_filter) if age >=0 else 0 for age in news_age]
    #         #     assert len(freshness_filter) == len(y_pred)
    #         #     y_pred = [a * b for a, b in zip(y_pred, freshness_filter)]

    #         y = [int(news[0].split('-')[1]) for news in minibatch['impressions']]
    
    #         try:
    #             auc = roc_auc_score(y, y_pred)
    #             mrr = mrr_score(y, y_pred)
    #             ndcg5 = ndcg_score(y, y_pred, 5)
    #             ndcg10 = ndcg_score(y, y_pred, 10)
    #         except ValueError:
    #             continue

    #         aucs.append(auc)
    #         mrrs.append(mrr)
    #         ndcg5s.append(ndcg5)
    #         ndcg10s.append(ndcg10)

    #     return np.mean(aucs), np.mean(mrrs), np.mean(ndcg5s), np.mean(ndcg10s)
    
    user2vector = {}
    for minibatch in tqdm(user_dataloader,
                        desc="Calculating vectors for users"):
        user_strings = minibatch["clicked_news_string"]
        if any(user_string not in user2vector for user_string in user_strings):
            clicked_news_vector = torch.stack([
                torch.stack([news2vector[x].to(device) for x in news_list],
                            dim=0) for news_list in minibatch["clicked_news"]
            ],dim=0).transpose(0, 1)
            # print('clicked_news_vector:',clicked_news_vector.shape)
            # print('clicked_news_vector:',clicked_news_vector)
            

            if model_name == 'LSTUR':
                user_vector = model.get_user_vector(
                    minibatch['user'], minibatch['clicked_news_length'],
                    clicked_news_vector)
            else:
                user_vector = model.get_user_vector(clicked_news_vector)
            for user, vector in zip(user_strings, user_vector):
                if user not in user2vector:
                    user2vector[user] = vector
                    
    for minibatch in tqdm(behaviors_dataloader,
                        desc="Calculating probabilities"):
        count += 1
        if count == max_count:
            break
        news_list = [news[0].split('-')[0] for news in minibatch['impressions']]

        # 사용 X (수명고려 안함함)
        # if config.test_filter != False:
        #     minibatch['time'] = pd.to_datetime(minibatch['time'])
        #     end_time = minibatch['time'].values[0] - np.timedelta64(config.lifetime,'h')
        #     if news_df.loc[news_list[0]]['publish_time'] < end_time:
        #         continue

        candidate_news_vector = torch.stack([news2vector[news] for news in news_list],dim=0)
        user_vector = user2vector[minibatch['clicked_news_string'][0]]
        click_probability = model.get_prediction(candidate_news_vector,
                                                user_vector)
        # print(click_probability.shape)
        y_pred = click_probability.tolist()
        

        # 사용 X (수명고려 안함함)
        # if config.test_filter != False:
        #     news_publish_time = news_df.loc[news_list]['publish_time'].to_list()
        #     news_age = [(minibatch['time'].values[0]-publish_time).total_seconds()/3600.00 for publish_time in news_publish_time ]
        #     freshness_filter = [sigmoid(36-age,config.test_filter) if age >=0 else 0 for age in news_age]

        #     assert len(freshness_filter) == len(y_pred)
        #     y_pred = [a * b for a, b in zip(y_pred, freshness_filter)]


        y = [int(news[0].split('-')[1]) for news in minibatch['impressions']]


        try:
            auc = roc_auc_score(y, y_pred)
            mrr = mrr_score(y, y_pred)
            ndcg5 = ndcg_score(y, y_pred, 5)
            ndcg10 = ndcg_score(y, y_pred, 10)
            acc = accurate_score(y, y_pred)
        except ValueError:
            continue

        aucs.append(auc)
        mrrs.append(mrr)
        ndcg5s.append(ndcg5)
        ndcg10s.append(ndcg10)
        accs.append(acc)

    return np.mean(aucs), np.mean(mrrs), np.mean(ndcg5s), np.mean(ndcg10s), np.mean(accs)


if __name__ == '__main__':
    # 모델 체크포인트를 불러와 평가를 수행
    # 평가 결과(AUC, MRR, nDCG@5, nDCG@10)를 출력
    
    print('Using device:', device)
    print(f'{config.data}Evaluating model {model_name}')
    # Don't need to load pretrained word/entity/context embedding
    # since it will be loaded from checkpoint later
    model = Model(config).to(device)
    from train import latest_checkpoint  # Avoid circular imports
    checkpoint_path = latest_checkpoint(path.join('./checkpoint', model_name))
    if checkpoint_path is None:
        print('No checkpoint file found!')
        exit()
    print(f"Load saved parameters in {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    auc, mrr, ndcg5, ndcg10, acc = evaluate(model, f'./data/preprocessed_data/{config.data}/test')
    print(
        f'AUC: {auc:.4f}\nMRR: {mrr:.4f}\nnDCG@5: {ndcg5:.4f}\nnDCG@10: {ndcg10:.4f}'
    )
