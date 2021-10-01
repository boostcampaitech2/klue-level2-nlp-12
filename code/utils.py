import os
import numpy as np
import pytz
import torch

from tqdm import tqdm
from datetime import datetime
from konlpy.tag import Mecab

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
