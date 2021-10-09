import os
import numpy as np
import pytz
import torch
import torch.nn as nn

from tqdm import tqdm
from datetime import datetime
# from konlpy.tag import Mecab
from sklearn.utils.validation import column_or_1d
from load_data import *

TRAIN_DATA_PATH = '/opt/ml/dataset/train/train.csv'

# AS_IS => 2021-09-30 14:44:46:57
# TO_BE => 2021-09-30-14:44:46:57
def korea_now():
    '''A function for calculating real-time based on Korea
        Raises:
            RuntimeError: Out of fuel

        Returns:
            cars: A car mileage
    '''


    try:
        now = datetime.now()
        korea = pytz.timezone("Asia/Seoul")
        korea_dt = korea.normalize(now.astimezone(korea))
        return '-'.join(str(korea_dt).split(".")[0].split())
    except:
        print('')



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


def get_entity_token_ids(input_ids: torch.tensor):
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
                                "per:children": "per:parents"},

        "cols": ['id', 'sentence', 'subject_entity', 'object_entity', 'label']
    }
    # 1. 훈련 데이터 불러와서 subject_entity 와 object_entitiy의 column name 만 바꾼다.
    data = load_data(TRAIN_DATA_PATH).rename(columns=config['change_entity'])
    # 2. subject_entity와 object_entity를 바꿀 수 있는 라벨만 남긴다.
    data = data[data['label'].isin(config['possible_label_list'])]
    # 3. column name을 정렬해준다. (1번에서 column name만 obj, sub 순으로 바꿔놨기 때문에 다시 sub, obj 순으로 정리하면 열 전체가 바뀐다.)
    data = data[config['cols']]
    # 4. 서로 반대되는 뜻을 가진 label을 바꿔준다.
    data = data.replace({'label': config['opposite_label_list']})

    return data


##########
# make new embedding module
##########
class RobertaEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, model):
        super(RobertaEmbeddings, self).__init__()
        self.tok_embed = model.embeddings.word_embeddings
        self.pos_embed = model.embeddings.position_embeddings
        self.ent_embed = nn.Embedding(2, 768)
        self.norm = model.embeddings.LayerNorm
        self.dropout = model.embeddings.dropout
        self.padding_idx = 1

    def forward(self, input_ids, token_type_ids=None, position_ids=None, inputs_embeds=None,
                past_key_values_length=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(self.padding_idx + 1, seq_length + self.padding_idx + 1, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        embedding = self.tok_embed(input_ids) + self.pos_embed(position_ids) + self.ent_embed(token_type_ids)
        norm = self.norm(embedding)
        return self.dropout(norm)