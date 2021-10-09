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
from error_handler import *
from ErrorMsg import *
from ErrorCode import *

TRAIN_DATA_PATH = '/opt/ml/dataset/train/train.csv'

def korea_now():
    '''A function for calculating real-time based on Korea

        Returns:
            time (datetime):
                AS_IS: 2021-09-30 14:44:46:57
                TO_BE: 2021-09-30-14:44:46:57
    '''
    now = datetime.now()
    korea = pytz.timezone("Asia/Seoul")
    korea_dt = korea.normalize(now.astimezone(korea))
    return '-'.join(str(korea_dt).split(".")[0].split())

def mkdir_model(t):
    '''A function for making directory to save best model

        Args:
            t (str): A folder name below best_model (top folder)

        Raises:
            CustomError: A folder name exists
    '''
    try:
        os.mkdir('./best_model/' + t)
    except:
        print(CustomError(ErrorCode.BAD_REQUEST, ErrorMsg.ALREADY_EXIST))

def get_entity_token_ids(input_ids):
    '''A function for getting entity ids from input ids

        Args:
            input_ids (torch.tensor): A word representaion vector from word tokens

        Returns:
            entity_ids (torch.tensor): A entity representation vector for entity tokens
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

def switch_sub_obj():
    '''A function for switching subject and object entities each other

        Returns:
            data (DataFrame): A dataframe made by switching subject - object entity positions if possible
    '''
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

    # call train data and change column name of subject and object entities
    data = load_data(TRAIN_DATA_PATH).rename(columns=config['change_entity'])
    # remain subject and object entities which are possibly switched
    data = data[data['label'].isin(config['possible_label_list'])]
    # re-order column name
    data = data[config['cols']]
    # switch labels that have different meaning
    data = data.replace({'label': config['opposite_label_list']})
    return data

class RobertaEmbeddings(nn.Module):
    '''
    A class for making new `embeddings` in **roberta model

    Args:
        model (nn.Module): A embeddings module from pretrained model offered by huggingface
    '''
    def __init__(self, model):
        super(RobertaEmbeddings, self).__init__()
        self.tok_embed = model.embeddings.word_embeddings       # from pretrinaed
        self.pos_embed = model.embeddings.position_embeddings   # from pretrinaed
        self.ent_embed = nn.Embedding(2, 768)                   # newly added for entity embedding
        self.norm = model.embeddings.LayerNorm                  # from pretrinaed
        self.dropout = model.embeddings.dropout                 # from pretrinaed
        self.padding_idx = 1

    def forward(self, input_ids, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=None):
        '''A forward function for including newly added entity embedding layer

        Args:
            input_ids (torch.tensor): A word representaion vector from word tokens
            token_type_ids (torch.tensor): A entity representation vector from entity tokens
            position_ids (torch.tensor): A position representation vector from position values

        Returns:
            output (torch.tensor): A 3D-tensors of output passing through `embeddings`
        '''
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(self.padding_idx + 1, seq_length + self.padding_idx + 1, dtype=torch.long,
                                        device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        embedding = self.tok_embed(input_ids) + self.pos_embed(position_ids) + self.ent_embed(token_type_ids)
        norm = self.norm(embedding)
        return self.dropout(norm)