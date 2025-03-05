import os
model_name = "NAML"


class BaseConfig():
    """
    General configurations appiled to all models


    모든 모델의 공통 및 개별 설정을 관리하는 파일
    모델 이름(model_name)과 관련 설정(예: num_epochs, candidate_type, batch_size 등)을 정의
    특정 모델별 설정도 클래스(LSTURConfig, NRMSConfig 등)로 제공됨됨

    """

    impre_ratio = 0.5         # 학습에 사용되는 노출 비율
    num_epochs = 5            # 학습 에포크 수
    early_stop_patience = 4   # 과적합을 방지하기 위한 early_stop 기준

    candidate_type = "random"                       # 후보 생성 방법 타입
    # candidate_type = "rev_current_log_pop"          # 후보 생성 방법 타입

    loss_function = "CEL"                           # 사용할 손실 함수 (교차 엔트로피 손실)
    negative_sampling_ratio = 4                     # 네거티브 샘플링 비율
    lifetime = 36                                   # 후보 뉴스 기사의 유지 기간 (개월 단위)
    numbering = "105"                               # 실험 또는 데이터셋을 식별하는 고유 번호
    data_folder = "experiment_data/baseline"        # 사용된 데이터셋 이름
    experiment_data = "behaviors_user1000_ns4_cdNone"      
    test_behaviors_file = "behaviors_user1000_ns20_cdNone.tsv" # 테스트 행동 데이터 파일 이름
    # test_filter = 0.05                              # 테스트 데이터 필터링 기준
    test_filter = False                             # 테스트 데이터 필터링 기준
    history_type = "random"                         # 고려되는 사용자 history 유형
    
    
    # 다른 후보 생성 접근 방식에 대한 설정 (주석 처리된 부분) 
    # candidate_type = "impre"
    our_type = "onetype"   #combine # onetype
    # loss_function = "CEL" #BPR_soft #BPR_sig #CEL



    num_batches_show_loss = 100  # Number of batchs to show loss
    # Number of batchs to check metrics on validation dataset
    # num_batches_validate = 1000
    batch_size = 50
    learning_rate = 0.0001
    num_workers = 0  # Number of workers for data loading
    num_clicked_news_a_user = 50  # Number of sampled click history for each user
    num_words_title = 20            # 각 뉴스 제목에서 고려되는 단어 수
    num_words_abstract = 50         # 각 뉴스 요약에서 고려되는 단어 수
    word_freq_threshold = 1         # 어휘에 포함될 단어의 빈도 임계값
    # entity_freq_threshold = 2
    # entity_confidence_threshold = 0.5
    dropout_probability = 0.2       # 정규화를 위한 드롭아웃 확률
    # Modify the following by the output of `src/dataprocess.py`
    num_words = 1 + 330899          # 어휘 크기 (고유 단어의 총 수)
    num_categories = 1 + 127        # 뉴스 카테고리 수
    # num_entities = 1 + 15587
    num_users = 1 + 229061          # 고유 사용자 수
    word_embedding_dim = 100        # 단어 임베딩 차원
    category_embedding_dim = 100    # 카테고리 임베딩 차원
    # Modify the following only if you use another dataset
    entity_embedding_dim = 100      # 엔티티 임베딩 차원
    # For additive attention
    query_vector_dim = 200          # attention 메커니즘에 사용되는 쿼리 벡터의 차원
    num_words_cat = 5               # 각 카테고리에 대해 고려되는 단어 수



class FIM_randomConfig(BaseConfig):
    dataset_attributes = {
        "news": ['title', 'category', 'subcategory'],   # 각 뉴스에 대해 고려되는 속성
        "record": []                                    # 사용자 상호작용 데이터 속성

    }
    # For CNN
    num_filters = 300               # CNN 층의 필터 수
    window_size = 3                 # 합성곱 윈도우 크기
    HDC_window_size = 3             # 고차원 합성곱 윈도우 크기
    HDC_filter_num = 150            # 고차원 합성곱 필터 수
    conv3D_filter_num_first = 32    # 첫 번째 3D 합성곱 층의 필터 수
    conv3D_kernel_size_first = 3    # 첫 번째 3D 합성곱 층의 커널 크기
    conv3D_filter_num_second = 16   # 두 번째 3D 합성곱 층의 필터 수
    conv3D_kernel_size_second = 3   # 두 번째 3D 합성곱 층의 커널 크기
    maxpooling3D_size = 3           # 3D 최대 풀링 윈도우 크기
    maxpooling3D_stride = 3         # 3D 최대 풀링 스트라이드

class FIMConfig(BaseConfig):
    dataset_attributes = {
        "news": ['title', 'category_word'],  # '카테고리 단어'를 포함한 뉴스 속성
        "record": []  # 사용자 상호작용 데이터 속성
    }
    # For CNN
    num_filters = 300
    window_size = 3
    HDC_window_size = 3
    HDC_filter_num = 150
    conv3D_filter_num_first = 32
    conv3D_kernel_size_first = 3
    conv3D_filter_num_second = 16
    conv3D_kernel_size_second = 3
    maxpooling3D_size = 3
    maxpooling3D_stride = 3

class NRMSConfig(BaseConfig):
    dataset_attributes = {"news": ['title'], "record": []}          # 뉴스와 사용자 기록 속성
    # For multi-head self-attention
    num_attention_heads = 10


class NAMLConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory', 'title', 'abstract'],   # 카테고리 및 요약을 포함한 뉴스 속성
        "record": []
    }
    # For CNN
    num_filters = 300
    window_size = 3


class LSTURConfig(BaseConfig):
    dataset_attributes = {
        "news": ['category', 'subcategory', 'title'],
        "record": ['user', 'clicked_news_length']                   # 클릭한 뉴스 길이를 포함한 사용자 속성
    }
    # For CNN
    num_filters = 300 # "원래 100이었는데 논문에 300이어서 수정함" (논문에 따라 100에서 300으로 변경)
    window_size = 3
    long_short_term_method = 'ini'  # 장기 및 단기 선호를 결합하는 방법
    # See paper for more detail
    assert long_short_term_method in ['ini', 'con']
    masking_probability = 0.5       # 사용자 행동을 마스킹할 확률


class DKNConfig(BaseConfig):
    dataset_attributes = {"news": ['title', 'title_entities'], "record": []}    # 제목 엔티티를 포함한 속성
    # For CNN
    num_filters = 50    # CNN 층의 필터 수
    window_sizes = [2, 3, 4]
    # TODO: currently context is not available
    use_context = False


class HiFiArkConfig(BaseConfig):
    dataset_attributes = {"news": ['title'], "record": []}
    # For CNN
    num_filters = 300
    window_size = 3
    num_pooling_heads = 5           # 풀링 층의 헤드 수
    regularizer_loss_weight = 0.1


class TANRConfig(BaseConfig):
    dataset_attributes = {"news": ['category', 'title'], "record": []}
    # For CNN
    num_filters = 300
    window_size = 3
    topic_classification_loss_weight = 0.1

