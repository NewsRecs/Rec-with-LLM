# 📖 Rec-with-LLM README



<br>


## 사전 실험 구조

```
├── README.md
└── Preliminary_experiment
     ├── App.jsx
     ├── index.jsx
     ├── api
     │     └── mandarinAPI.js
     ├── data
     │     ├── history
     │     │     ├── 20170101
     │     │     ├── 20170102
     │     │     ├── 20170103
     │     │     └── news.tsv
     │     ├── train
     │     │     ├── 20170104
     │     │     ├── 20170105
     │     │     ├── 20170106
     │     │     ├── behaviors.tsv
     │     │     └── news.tsv
     │     └── user.tsv
     ├── gpt_finetuning_data
     │     ├── train_negative.jsonl
     │     ├── train_positive.jsonl
     │     ├── val_negative.jsonl
     │     └── val_positive.jsonl
     ├── result
     │     ├── error_detect
     │     │     ├── negative_detected.txt
     │     │     ├── negative_finetuned_detected.txt
     │     │     ├── positive_detected.txt
     │     │     └── positive_finetuned_detected.txt
     │     ├── experiment_result
     │     │     ├── negative_metrics.txt
     │     │     ├── negative_finetuned_metrics.txt
     │     │     ├── positive_metrics.txt
     │     │     └── positive_finetuned_metrics.txt
     │     └── gpt_result
     │           ├── negative.txt
     │           ├── negative_finetuned.txt
     │           ├── positive.txt
     │           └── positive_finetuned.txt
     ├── .env
     ├── 0. debugging.ipynb
     ├── 0. prompt_tokens_check.ipynb
     ├── 1. creat_prompts.ipynb
     ├── 2. generate_json.ipynb
     ├── 3. experiment.ipynb
     └── 4. create_metrics
```

<br>

## 폴더 설명
### data

- **user data 및 behaviors data가 담긴 폴더**

    
### gpt_finetuning_data

- **GPT API Fine-tuning에 사용할 JSONL 파일이 담긴 폴더**


### result

- **error_detect**
    - 실험 결과에 오류가 있는 USER 목록을 담은 폴더
- **experiment_result**
    - 실험 결과 Metric(nDCG@5)을 담은 폴더
- **gpt_result**
    - 실험 결과 폴더


<br>


## 파일 설명

#### .env
- GPT API KEY가 담긴 파일

#### 0. debugging.ipynb
- 실험 결과 파일에서 각 USER 마다 오류를 찾는 파일

#### 0. prompt_tokens_check.ipynb
- user prompt의 token을 계산하는 파일
  
#### 1. create_prompts.ipynb
- 각 user의 prompt를 생성하는 파일

#### 2. generate_json.ipynb
- GPT API Fine-tuning에 사용할 JSONL Data을 생성하는 파일
  
#### 3. experiment.ipynb
- GPT API 를 사용하여 실험하는 파일
  
#### 4. create_metrics.ipynb
- 실험 결과 파일의 metrics(nDCG@5)를 계산하는 파일

<br>
