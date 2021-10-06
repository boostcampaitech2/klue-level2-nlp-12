import pickle as pickle
import os
import pandas as pd
import torch

from sklearn.model_selection import train_test_split

class RE_Dataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {
            key: val[idx].clone().detach() for key, val in self.pair_dataset.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels

def preprocessing_test_dataset(dataset):
    """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    for i, j in zip(dataset["subject_entity"], dataset["object_entity"]):
        i = eval(i)["word"]
        j = eval(j)["word"]

        subject_entity.append(i)
        object_entity.append(j)

    test_dataset = pd.DataFrame(
        {
            "id": dataset["id"],
            "sentence": dataset["sentence"],
            "subject_entity": subject_entity,
            "object_entity": object_entity,
            "label": dataset["label"],
        }
    )
    
    return test_dataset

def load_test_data(dataset_dir):
    """csv 파일을 경로에 맡게 불러 옵니다."""
    pd_dataset = pd.read_csv(dataset_dir)
    test_dataset = preprocessing_test_dataset(pd_dataset)

    return test_dataset

def preprocessing_train_dataset(dataset):
    """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    for i, j in zip(dataset["subject_entity"], dataset["object_entity"]):
        i = eval(i)["word"]
        j = eval(j)["word"]

        subject_entity.append(i)
        object_entity.append(j)
    out_dataset = pd.DataFrame(
        {
            "id": dataset["id"],
            "sentence": dataset["sentence"],
            "subject_entity": subject_entity,
            "object_entity": object_entity,
            "label": dataset["label"],
        }
    )
    # ['sentence', 'subject_entity', 'object_entity'] 패턴의 중복 행 제거
    out_dataset = out_dataset.drop_duplicates(['sentence', 'subject_entity', 'object_entity'], keep='first')

    train_dataset, dev_dataset = train_test_split(out_dataset, test_size=0.2, stratify=out_dataset['label'], random_state=42)
    
    return train_dataset, dev_dataset


def load_train_data(dataset_dir):
    """csv 파일을 경로에 맡게 불러 옵니다."""
    pd_dataset = pd.read_csv(dataset_dir)
    train_dataset, dev_dataset = preprocessing_train_dataset(pd_dataset)

    return train_dataset, dev_dataset


def preprocessing_dataset(dataset):
    """처음 불러온 csv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다."""
    subject_entity = []
    object_entity = []
    subject_entity_type = []
    object_entity_type = []

    for sub, obj in zip(dataset["subject_entity"], dataset["object_entity"]):
        i = eval(sub)["word"]
        j = eval(obj)["word"]
        k = eval(sub)["type"]
        l = eval(obj)["type"]

        subject_entity.append(i)
        object_entity.append(j)
        subject_entity_type.append(k)
        object_entity_type.append(l)

    out_dataset = pd.DataFrame(
        {
            "id": dataset["id"],
            "sentence": dataset["sentence"],
            "subject_entity": subject_entity,
            "object_entity": object_entity,
            "subject_entity_type": subject_entity_type,
            "object_entity_type": object_entity_type,
            "label": dataset["label"],
        }
    )
    # ['sentence', 'subject_entity', 'object_entity'] 패턴의 중복 행 제거
    out_dataset = out_dataset.drop_duplicates(['sentence', 'subject_entity', 'object_entity'], keep='first')
    
    return out_dataset


def load_data(dataset_dir):
    """csv 파일을 경로에 맡게 불러 옵니다."""
    pd_dataset = pd.read_csv(dataset_dir)
    out_dataset = preprocessing_dataset(pd_dataset)

    return out_dataset


# https://huggingface.co/transformers/internal/tokenization_utils.html
def tokenized_dataset(dataset, tokenizer, token_type, model):
    """tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []

    if token_type=='default':
        for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
            temp = ""
            temp = e01 + "[SEP]" + e02
            concat_entity.append(temp)
    if token_type=='swap_entity':
        for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
            temp = ""
            temp = e02 + "[SEP]" + e01
            concat_entity.append(temp)
    elif token_type=='sentence_entity':
        for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
            temp = ""
            temp = f'{e01}과 {e02}는 어떤 관계일까?' #f-string
            concat_entity.append(temp)
    elif token_type=='punct_typed_entity':
        for e01, e02, t01, t02 in zip(dataset["subject_entity"], dataset["object_entity"], dataset["subject_entity_type"], dataset["object_entity_type"]):
            temp = ""
            temp = f'@ * {t01} * {e01} @ # ^ {t02} ^ {e02} #'
            concat_entity.append(temp)

    tokenized_sentences = None
    
    if model=='klue':
        tokenized_sentences = tokenizer(
            concat_entity,
            list(dataset["sentence"]),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
            add_special_tokens=True,
            #return_token_type_ids=False,
        )
    else:
        tokenized_sentences = tokenizer(
            concat_entity,
            list(dataset["sentence"]),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
            add_special_tokens=True,
            return_token_type_ids=False,
        )

    return tokenized_sentences
