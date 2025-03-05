from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from dataset import BaseDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from config import model_name
from tqdm import tqdm
import os
from pathlib import Path
from evaluate import evaluate
import importlib
import datetime
import copy

try:
    # getattr 함수로 model.model_name 모듈 안에서 이름이 model_nam
    # e인 클래스(__init__.py에서서)를 가져옴
    Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
except AttributeError:
    print(f"{model_name} not included!")
    exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EarlyStopping:
    def __init__(self, patience=config.early_stop_patience):
        # 조기 종료 기준을 위한 patience 설정
        self.patience = patience
        self.counter = 0
        self.best_loss = np.Inf

    def __call__(self, val_loss):
        """
        성능이 개선되지 않으면 카운터를 증가시키고, 개선되었을 때는 카운터 초기화
        """
        if val_loss < self.best_loss:
            early_stop = False
            get_better = True
            self.counter = 0
            self.best_loss = val_loss
        else:
            get_better = False
            self.counter += 1
            if self.counter >= self.patience:
                early_stop = True
            else:
                early_stop = False
        if np.isnan(val_loss):
            early_stop = True
        return early_stop, get_better


def latest_checkpoint(directory):
    # 가장 최신 체크포인트 찾기 (특정 형식에 맞는 파일 찾기)
    if not os.path.exists(directory):
        return None
    all_checkpoints = {
        int(x.split('.')[0].split('-')[1]): x
        for x in os.listdir(directory)
        if (x.split('.')[0].split('-')[2] == config.candidate_type)
        if (x.split('.')[0].split('-')[3] == config.our_type)
        if (x.split('.')[0].split('-')[4] == config.loss_function)
    }
    if not all_checkpoints:
        return None
    return os.path.join(directory,
                        all_checkpoints[max(all_checkpoints.keys())])

def time_since(since):
    """
    Format elapsed time string.
    """
    now = time.time()
    elapsed_time = now - since
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


def train():
    """
    모델 학습을 위한 train 함수 정의
    """

    # 결과 파일 이름 설정
    result_file = f"./results/{model_name}/{config.experiment_data}"
    result_file = f"{result_file}.txt"
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')


    # 사전 학습된 임베딩 불러오기
    try:
        pretrained_word_embedding = torch.from_numpy(
             np.load(f'{config.data_folder}/pretrained_word_embedding.npy')).float()
    except FileNotFoundError:
        pretrained_word_embedding = None


    if model_name == 'DKN':
        # DKN 모델의 경우 추가 임베딩 불러오기 (entity 및 context 임베딩)
        try:
            pretrained_entity_embedding = torch.from_numpy(
                np.load(
                    f'{config.data_folder}/pretrained_entity_embedding.npy')).float()
        except FileNotFoundError:
            pretrained_entity_embedding = None

        try:
            pretrained_context_embedding = torch.from_numpy(
                np.load(
                    f'{config.data_folder}/pretrained_context_embedding.npy')).float()
        except FileNotFoundError:
            pretrained_context_embedding = None

        # 모델 초기화
        model = Model(config, pretrained_word_embedding,
                      pretrained_entity_embedding,
                      pretrained_context_embedding).to(device)
    elif model_name == 'Exp1':
        # Exp1 모델의 경우 앙상블 사용
        models = nn.ModuleList([
            Model(config, pretrained_word_embedding).to(device)
            for _ in range(config.ensemble_factor)
        ])
    elif model_name == 'Exp2':
        model = Model(config).to(device)
    else:
        model = Model(config, pretrained_word_embedding).to(device)


    if model_name != 'Exp1':
        print(model)
    else:
        print(models[0])

 
    
    dataset = BaseDataset(f'{config.data_folder}/train/{config.experiment_data}.tsv', # behaviors_path
                          f'{config.data_folder}/news_parsed.tsv',                        # news_path
                          f'{config.data_folder}/roberta')                          # roberta_embedding_dir

    print(f"Load training dataset with size {len(dataset)}.")

    dataloader = iter(
        DataLoader(dataset,
                   batch_size=config.batch_size,
                   shuffle=True,
                   num_workers=config.num_workers,
                   drop_last=True,
                   pin_memory=True))
    


    # 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                    lr=config.learning_rate)
    print(f"Loss function:{config.loss_function}, NS Type: {config.candidate_type}_{config.our_type}")

    start_time = time.time()
    loss_full = []
    exhaustion_count = 0
    step = 0
    early_stopping = EarlyStopping()

    # 체크포인트 및 결과 디렉토리 생성
    checkpoint_dir = os.path.join('./checkpoint', model_name)
    result_dir = os.path.join('./results', model_name)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(result_dir).mkdir(parents=True, exist_ok=True)

    # 가장 최근 체크포인트 불러오기 (없으면 None)
    checkpoint_path = latest_checkpoint(checkpoint_dir)
    # checkpoint_path = None

    epoch_result = []
    if checkpoint_path is not None:
        # 체크포인트에서 파라미터 불러오기
        print(f"Load saved parameters in {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        early_stopping(checkpoint['early_stop_value'])
        step = checkpoint['step']
        exhaustion_count = checkpoint['exhaustion_count']
        epoch_result = [x.split(' ') for x in checkpoint['epoch_result'].split('\n')]
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.train()

    # 모델 학습 루프
    for i in tqdm(range(
            1,
            config.num_epochs * len(dataset) // config.batch_size + 1),
                  desc="Training"):
        try:
            minibatch = next(dataloader)
        except StopIteration:
            # 데이터셋이 끝났을 때, 다시 반복 설정
            exhaustion_count += 1

            # 검증 데이터 평가 및 결과 저장
            (model if model_name != 'Exp1' else models[0]).eval()
            val_auc, val_mrr, val_ndcg5, val_ndcg10 = evaluate(
                model if model_name != 'Exp1' else models[0], f'{config.data_folder}')
            (model if model_name != 'Exp1' else models[0]).train()

            tqdm.write(
                f"Time {time_since(start_time)}, batches {i}\nvalidation AUC: {val_auc:.4f}, validation MRR: {val_mrr:.4f}, validation nDCG@5: {val_ndcg5:.4f}, validation nDCG@10: {val_ndcg10:.4f}, "
            )
            print()
            print('┌─────────────┐')
            print(f'│{exhaustion_count} Epoch Done!│')
            print('└─────────────┘')
            print()

            # 소수점 4자리로 반올림
            rounded_values = [round(val_auc, 4), round(val_mrr, 4), round(val_ndcg5, 4), round(val_ndcg10, 4)]
            epoch_result.append([exhaustion_count] + rounded_values)
            
            # 텍스트 파일로 저장
            with open(result_file, 'w') as wf:
                wf.write("Epoch\tValidation AUC\tValidation MRR\tValidation nDCG@5\tValidation nDCG@10\n")
                for row in epoch_result:
                    wf.write(f"{row[0]}\t{row[1]:.4f}\t{row[2]:.4f}\t{row[3]:.4f}\t{row[4]:.4f}\n")


            # epoch_result.append([str(val_auc),str(val_mrr),str(val_ndcg5),str(val_ndcg10)])
            # with open(result_file,'w') as wf:
            #     line = '\n'.join([ ' '.join(x) for x in epoch_result])
            #     wf.write(line)

            early_stop, get_better = early_stopping(-sum([val_auc,val_mrr,val_ndcg5,val_ndcg10]))
            if early_stop:
                tqdm.write(f'{exhaustion_count} Epoch Done! Early stop.')
                break

            if exhaustion_count == config.num_epochs:
                break

            # 데이터 로더 다시 설정
            dataloader = iter(
                DataLoader(dataset,
                           batch_size=config.batch_size,
                           shuffle=True,
                           num_workers=config.num_workers,
                           drop_last=True,
                           pin_memory=True))
            minibatch = next(dataloader)

        step += 1

        # 모델 예측 및 손실 계산 (forward() 메서드 실행)
        if model_name == 'LSTUR':
            y_pred = model(minibatch["user"], minibatch["clicked_news_length"],
                           minibatch["candidate_news"],
                           minibatch["clicked_news"])
        elif model_name == 'HiFiArk':
            y_pred, regularizer_loss = model(minibatch["candidate_news"],
                                             minibatch["clicked_news"])
        elif model_name == 'TANR':
            y_pred, topic_classification_loss = model(
                minibatch["candidate_news"], minibatch["clicked_news"])
        else:
            y_pred = model(minibatch["candidate_news"], minibatch["clicked_news"])

        y_true = torch.zeros(len(y_pred)).long().to(device)
        loss = criterion(y_pred, y_true)
        if model_name == 'HiFiArk':
            loss += config.regularizer_loss_weight * regularizer_loss
        loss_full.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 손실 값을 일정 주기로 출력
        if i % config.num_batches_show_loss == 0:
            tqdm.write(
                f"Time {time_since(start_time)}, batches {i}, current loss {loss.item():.4f}, average loss: {np.mean(loss_full):.4f}, latest average loss: {np.mean(loss_full[-256:]):.4f}"
            )
            if np.isnan(loss.item()):
                break

    # 마지막 평가 및 결과 저장
    (model if model_name != 'Exp1' else models[0]).eval()
    val_auc, val_mrr, val_ndcg5, val_ndcg10 = evaluate(
        model if model_name != 'Exp1' else models[0], f'{config.data_folder}')
    

    if [str(val_auc), str(val_mrr), str(val_ndcg5), str(val_ndcg10)] not in epoch_result:
        # 소수점 4자리로 반올림
        rounded_values = [round(val_auc, 4), round(val_mrr, 4), round(val_ndcg5, 4), round(val_ndcg10, 4)]
        epoch_result.append([exhaustion_count] + rounded_values)
        
        # 텍스트 파일로 저장
        with open(result_file, 'w') as wf:
            wf.write("Epoch\tValidation AUC\tValidation MRR\tValidation nDCG@5\tValidation nDCG@10\n")
            for row in epoch_result:
                wf.write(f"{row[0]}\t{row[1]:.4f}\t{row[2]:.4f}\t{row[3]:.4f}\t{row[4]:.4f}\n")






if __name__ == '__main__':
    print('Using device:', device)
    print(f'Training model {model_name}')
    train()
