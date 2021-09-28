import pickle as pickle
import os
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer

from pandas import DataFrame
from sklearn.model_selection import StratifiedKFold
import numpy as np


class RE_Dataset(torch.utils.data.Dataset):
  """ Dataset 구성을 위한 class."""
  def __init__(self, pair_dataset, labels):
    self.pair_dataset = pair_dataset
    self.labels = labels

    print('--- Pair Dataset ---')
    print(pair_dataset)

  def __getitem__(self, idx):
    item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

def preprocessing_test_dataset(dataset:pd.DataFrame):
  '''
  A Preprocessing function to convert original test dataset to useful one

  :param dataset (DataFrame): an original test dataset from train.csv
  :return:
  '''
  subject_entity = []
  object_entity = []
  for i, j in zip(dataset['subject_entity'], dataset['object_entity']):
    i = i[1:-1].split(',')[0].split(':')[1]  # 비틀즈
    j = j[1:-1].split(',')[0].split(':')[1]  # 조지 해리슨

    subject_entity.append(i)
    object_entity.append(j)
  out_dataset = pd.DataFrame({'id': dataset['id'], 'sentence': dataset['sentence'], 'subject_entity': subject_entity,
                              'object_entity': object_entity, 'label': dataset['label']})
  return out_dataset

def preprocessing_dataset(dataset:pd.DataFrame):
  '''
  A Preprocessing function to convert original dataset to useful one

  :param dataset (DataFrame): an original train dataset from train.csv
  :return:
  '''
  subject_entity = []
  object_entity = []
  for i,j in zip(dataset['subject_entity'], dataset['object_entity']):
    i = i[1:-1].split(',')[0].split(':')[1] # 비틀즈
    j = j[1:-1].split(',')[0].split(':')[1] # 조지 해리슨

    subject_entity.append(i)
    object_entity.append(j)
  out_dataset = pd.DataFrame({'id':dataset['id'], 'sentence':dataset['sentence'],'subject_entity':subject_entity,'object_entity':object_entity,'label':dataset['label']})

  # split dataset into train, valid
  train_set, val_set = train_test_split(out_dataset, test_size=0.2, stratify=dataset['label'], random_state=42)

  print('--- Train Set Length ---')
  print(len(train_set))

  print('--- Val Set Length ---')
  print(len(val_set))
  return train_set, val_set

def load_test_data(dataset_dir:str):
  '''
  Load original dataset from test.csv

  :param dataset_dir (str): a path of test.csv
  :return:
  '''
  pd_dataset = pd.read_csv(dataset_dir)
  test_set = preprocessing_test_dataset(pd_dataset)
  return test_set

def load_data(dataset_dir:str):
  '''
  Load original dataset from train.csv

  :param dataset_dir (str): a path of train.csv
  :return:
  '''
  pd_dataset = pd.read_csv(dataset_dir)
  train_set, val_set = preprocessing_dataset(pd_dataset)
  return train_set, val_set

def tokenized_dataset(dataset, tokenizer):
    """ tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset['subject_entity'], dataset['object_entity']):
        temp = ''
        temp = e01 + '[SEP]' + e02

        # 주어 + 목적어 pair
        concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        # max_length=46,
        add_special_tokens=True,
        return_token_type_ids=False
    )

    print('--- Decode Tokenized Sentences ---')
    print(tokenizer.decode(tokenized_sentences['input_ids'][0]))
    return tokenized_sentences


def make_stratifiedkfold(raw_train: DataFrame, raw_train_classes: list, n_splits: int, shuffle: bool, seed: int):
    """ #### raw_train dataset을 stratifiedkfold로 나눠 주는 함수
    raw data -> stratify 한 fklod로 만들어줌

    Example:

        >>> flods = make_stratifiedkfold(조건에 맞춰 arg 지정)
        >>> for fold, (trn_idx, val_idx) in enumerate(folds):
        >>>     train_df = raw_train.loc[trn_idx, :].reset_index(drop=True)
        >>>     valid_df = raw_train.loc[val_idx, :].reset_index(drop=True)

        >>>     train_dataset = DataSet(train_df)
        >>>     valid_dataset = DataSet(valid_df)
    Args:
        raw_train (DataFrame): origin `train.csv` dataframe
        raw_train_classes (list): raw data class(label)
        n_splits (int): fold 갯수
        shuffle (bool): 섞을 것인지 
        seed (int): Random seed

    Returns:
        folds (Generator): (train idx, valid idx) 쌍을 생성
    """
    folds = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=seed
    ).split(np.arange(raw_train.shape[0]), raw_train_classes)
    return folds


def make_train_df(raw_train: DataFrame, train_index: int, valid_index: int):
    """
    `make_stratifiedkfold()` 의 (train idx, valid idx) 쌍에 따라 Stratified 한 train_df, valid_df 생성

    Args:
        raw_train (DataFrame): origin `train.csv` dataframe
        train_index (int): `make_stratifiedkfold()` 의 train idx
        valid_index (int): `make_stratifiedkfold()` 의 valid idx

    Returns:
        train_df, valid_df: Dataset에 넣을 Dataframe
    """
    train_df = raw_train.loc[train_index, :].reset_index(drop=True)
    valid_df = raw_train.loc[valid_index, :].reset_index(drop=True)
    return train_df, valid_df
