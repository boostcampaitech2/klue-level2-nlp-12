import os
import numpy as np
import pytz
import torch

from transformers import AutoModelForSequenceClassification
from tqdm import tqdm
from datetime import datetime
from konlpy.tag import Mecab
from sklearn.utils.validation import column_or_1d
from load_data import *
from ray import tune

TRAIN_DATA_PATH = '/opt/ml/dataset/train/train.csv'

# AS_IS => 2021-09-30 14:44:46:57
# TO_BE => 2021-09-30-14:44:46:57
def korea_now():
    """
    현재시간을 알려주는 함수
    혹시 파일생성시 파일명 안 겹치게 저장하려고 만들었습니다
    """
    now = datetime.now()
    korea = pytz.timezone("Asia/Seoul")
    korea_dt = korea.normalize(now.astimezone(korea))
    return '-'.join(str(korea_dt).split(".")[0].split())

def mkdir_model(t):
    '''
    A folder to contain trained model

    :param t (datetime) : Now
    :return:
    '''
    try:
        os.mkdir('./best_model/' + t)
    except:
        print('[ Not Found Path ] 모델을 저장할 폴더를 생성할 수 없습니다.')
        exit()

def get_entity_token_ids(input_ids:torch.tensor):
    '''
    A function for getting entity ids from input ids

    :param input_ids (tensor): input ids from tokenized words
    :return: entity_ids (tensor): [0, 0, 1, 0, 0, 0, 1, 0 ...]
    '''
    input_ids = input_ids.numpy()
    flag = False
    entity_ids = []
    for i in tqdm(input_ids):
        tmp = []
        for j in i:
            if j == 32000:
                flag = True
                tmp += [0]
            elif j == 32001:
                flag = False
                tmp += [0]
            else:
                if flag:
                    tmp += [1]
                else:
                    tmp += [0]
        entity_ids.append(tmp)

    entity_ids = torch.tensor(entity_ids)
    print('--- Entity Embedding Ids ---')
    print(entity_ids[0])
    print(entity_ids.shape)
    return entity_ids

# def get_mecab_tokenized_result(sentence):
#
#     pass

def switch_sub_obj():
    config = {
        "change_entity": {"subject_entity": "object_entity", "object_entity": "subject_entity"},

        "possible_label_list": ['no_relation', 'org:members', 'org:alternate_names',
                              'per:children', 'per:alternate_names', 'per:other_family', 
                              'per:colleagues', 'per:siblings', 'per:spouse',
                              'org:member_of', 'per:parents'],
        
        "opposite_label_list": {"org:member_of": "org:members", 
                          "org:members": "org:member_of", 
                          "per:parents": "per:children", 
                          "per:children": "per:parents" },
        
        "cols": ['id', 'sentence', 'subject_entity', 'object_entity', 'label']
    }
    # 1. 훈련 데이터 불러와서 subject_entity 와 object_entitiy의 column name 만 바꾼다.
    data = load_data(TRAIN_DATA_PATH).rename(columns=config['change_entity'])
    # 2. subject_entity와 object_entity를 바꿀 수 있는 라벨만 남긴다.
    data = data[data['label'].isin(config['possible_label_list'])]
    # 3. column name을 정렬해준다. (1번에서 column name만 obj, sub 순으로 바꿔놨기 때문에 다시 sub, obj 순으로 정리하면 열 전체가 바뀐다.)
    data = data[config['cols']]
    # 4. 서로 반대되는 뜻을 가진 label을 바꿔준다.
    data = data.replace({'label':config['opposite_label_list']})

    return data

def ray_hp_space(trial):
    '''
    A function for searching best hyperparameters by ray tune

    :param:
    :return:
    '''
    return {
        "learning_rate": tune.loguniform(5e-4, 5e-6),
        "num_train_epochs": tune.choice(range(1, 6)),
        "per_device_train_batch_size": tune.choice([64, 128, 256]),
        "seed": tune.choice(range(1, 42)),
    }

def optuna_hp_space(trial):
    '''
    A function for searching best hyperparameters by optuna

    :param:
    :return:
    '''
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-6, 5e-4, log=True),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 6),
        "seed": trial.suggest_int("seed", 1, 42),
    }