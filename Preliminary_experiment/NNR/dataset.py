from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval
from os import path
import numpy as np
from config import model_name
import importlib
import torch
import random

# 설정을 동적으로 가져오기 위해 사용
try:
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()


class BaseDataset(Dataset):
    """
    추천 시스템 훈련을 위한 데이터를 로드하고 전처리하는 역할을 합니다.
    BaseDataset 클래스는 사용자 행동(behaviors) 및 뉴스 데이터(news)를 로드합니다.
    모델에 필요한 입력 형식(예: 클릭 뉴스 히스토리, 후보 뉴스 등)으로 변환하며, 로버타 임베딩도 지원합니다.
    """
    
    def __init__(self, behaviors_path, news_path, roberta_embedding_dir):
        # model이 Exp2가 아니면 roberta embedding은 필요 없음
        """
        1. behavior dataset
        2. news dataset
        3. news_id2int
        4. news2dict
        5. padding
        """

        super(BaseDataset, self).__init__()

        # 설정 파일에서 정의한 데이터 속성들이 올바른지 확인
        assert all(attribute in [
            'category', 'subcategory', 'title', 'abstract', 'title_entities',"category_word",
            'abstract_entities', 'title_roberta', 'title_mask_roberta',
            'abstract_roberta', 'abstract_mask_roberta'
        ] for attribute in config.dataset_attributes['news'])

        assert all(attribute in ['user', 'clicked_news_length']
                   for attribute in config.dataset_attributes['record'])
        

        # 행동 데이터 파일을 읽어와 DataFrame으로 저장
        self.behaviors_parsed = pd.read_table(behaviors_path)
        # print(self.behaviors_parsed.columns)

        # 뉴스 데이터 파일을 읽어와 필요한 컬럼만 선택하여 DataFrame으로 저장
        self.news_parsed = pd.read_table(
            news_path,
            index_col='id',
            usecols=['id'] + config.dataset_attributes['news'],
            converters={
                attribute: literal_eval
                for attribute in set(config.dataset_attributes['news']) & set([
                    'title', 'abstract', 'title_entities', 'abstract_entities',"category_word",
                    'title_roberta', 'title_mask_roberta', 'abstract_roberta',
                    'abstract_mask_roberta'
                ])
            })
        
        # 뉴스 ID를 인덱스로 변환하여 저장
        self.news_id2int = {x: i for i, x in enumerate(self.news_parsed.index)}

        # 뉴스 정보를 딕셔너리 형태로 저장하고 텐서로 변환
        self.news2dict = self.news_parsed.to_dict('index')
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                self.news2dict[key1][key2] = torch.tensor(
                    self.news2dict[key1][key2])
                
        # 패딩 정보를 설정
        padding_all = {
            'category': 0,
            'subcategory': 0, "category_word" : [0] * (config.negative_sampling_ratio + 1),
            'title': [0] * config.num_words_title,
            'abstract': [0] * config.num_words_abstract,
            'title_entities': [0] * config.num_words_title,
            'abstract_entities': [0] * config.num_words_abstract,
            'title_roberta': [0] * config.num_words_title,
            'title_mask_roberta': [0] * config.num_words_title,
            'abstract_roberta': [0] * config.num_words_abstract,
            'abstract_mask_roberta': [0] * config.num_words_abstract
        }
        for key in padding_all.keys():
            padding_all[key] = torch.tensor(padding_all[key])

        # 실제 사용될 패딩 정보만 필터링하여 저장
        self.padding = {
            k: v
            for k, v in padding_all.items()
            if k in config.dataset_attributes['news']
        }

        # 특정 모델 이름과 설정에 따른 로버타 임베딩 처리
        if model_name == 'Exp2' and not config.fine_tune:
            if config.roberta_level == 'word':
                self.roberta_embedding = {
                    k: torch.from_numpy(
                        np.load(
                            path.join(roberta_embedding_dir,
                                      f'{k}_last_hidden_state.npy'))).float()
                    for k in set(config.dataset_attributes['news'])
                    & set(['title', 'abstract'])
                }
                name2length = {
                    'title': config.num_words_title,
                    'abstract': config.num_words_abstract
                }
                for k in set(config.dataset_attributes['news']) & set(
                    ['title', 'abstract']):
                    self.padding[k] = torch.zeros((name2length[k], 768))

            elif config.roberta_level == 'sentence':
                self.roberta_embedding = {
                    k: torch.from_numpy(
                        np.load(
                            path.join(roberta_embedding_dir,
                                      f'{k}_pooler_output.npy'))).float()
                    for k in set(config.dataset_attributes['news'])
                    & set(['title', 'abstract'])
                }
                for k in set(config.dataset_attributes['news']) & set(
                    ['title', 'abstract']):
                    self.padding[k] = torch.zeros(768)

    # 뉴스 ID를 딕셔너리로 변환하는 메소드
    def _news2dict(self, id):
        ret = self.news2dict[id]
        if model_name == 'Exp2' and not config.fine_tune:
            for k in set(config.dataset_attributes['news']) & set(
                ['title', 'abstract']):
                ret[k] = self.roberta_embedding[k][self.news_id2int[id]]
        return ret

    # 데이터셋의 길이를 반환하는 메소드
    def __len__(self):
        return len(self.behaviors_parsed)

    # 인덱스를 통해 특정 샘플을 가져오는 메소드
    def __getitem__(self, idx):
        item = {}
        row = self.behaviors_parsed.iloc[idx]

        # 사용자 정보를 추가
        if 'user' in config.dataset_attributes['record']:
            item['user'] = row.user

        # 클릭한 뉴스 정보를 추가
        item["clicked"] = list(map(int, row.clicked.split()))
        candidate_type = config.candidate_type

        # # 후보 뉴스 설정에 따른 처리
        # if candidate_type == "rev_current_log_pop":
        #     item["candidate_news"] = [self._news2dict(x) for x in row.candidate_news.split()]
        item["candidate_news"] = [self._news2dict(x) for x in row.candidate_news.split()]

        # 클릭한 뉴스 히스토리를 추가하고 설정에 따라 랜덤 셔플
        clicked_news_split = row.clicked_news.split() 
        if config.history_type == 'random':
            try:
                random.shuffle(clicked_news_split)
            except:
                pass
        
        # 클릭한 뉴스 목록을 설정된 수만큼 추가
        item["clicked_news"] = [
            self._news2dict(x)
            for x in clicked_news_split[-config.num_clicked_news_a_user:]
        ]
                
        # 클릭한 뉴스 길이를 추가
        if 'clicked_news_length' in config.dataset_attributes['record']:
            item['clicked_news_length'] = len(item["clicked_news"])

        # 클릭한 뉴스 수가 설정된 수보다 적으면 패딩 추가
        repeated_times = config.num_clicked_news_a_user - \
            len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = [self.padding
                                ] * repeated_times + item["clicked_news"]

        return item
