import pickle as pickle
import os
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer

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
