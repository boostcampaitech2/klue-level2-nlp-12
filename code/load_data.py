import pickle as pickle
import os
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, BertTokenizer, RobertaTokenizer, BertTokenizerFast, Trainer

from pandas import DataFrame
from sklearn.model_selection import StratifiedKFold
import numpy as np
from torch.utils.data import DataLoader
from utils import *


class RE_Dataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 class."""

    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

        print("--- Pair Dataset ---")
        print(pair_dataset)

    def __getitem__(self, idx):
        item = {
            key: val[idx].clone().detach() for key, val in self.pair_dataset.items()
        }
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def preprocessing_test_dataset(dataset: pd.DataFrame):
    """
    A Preprocessing function to convert original test dataset to useful one

    :param dataset (DataFrame): an original test dataset from train.csv
    :return:
    """
    subject_entity = []
    object_entity = []
    for i, j in zip(dataset["subject_entity"], dataset["object_entity"]):
        i = eval(i)["word"]  # 비틀즈
        j = eval(j)["word"]  # 조지 해리슨

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
    return out_dataset


def preprocessing_dataset(dataset: pd.DataFrame):
    """
    A Preprocessing function to convert original dataset to useful one

    :param dataset (DataFrame): an original train dataset from train.csv
    :return:
    """
    subject_entity = []
    object_entity = []
    for i, j in zip(dataset["subject_entity"], dataset["object_entity"]):
        i = eval(i)["word"]  # 비틀즈
        j = eval(j)["word"]  # 조지 해리슨

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

    # split dataset into train, valid
    # train_set, val_set = train_test_split(
    #     out_dataset, test_size=0.2, stratify=dataset["label"], random_state=42
    # )
    #
    # print("--- Train Set Length ---")
    # print(len(train_set))
    #
    print("--- Data Set Length ---")
    print(len(out_dataset))
    return out_dataset


def load_test_data(dataset_dir: str):
    """
    Load original dataset from test.csv

    :param dataset_dir (str): a path of test.csv
    :return:
    """
    pd_dataset = pd.read_csv(dataset_dir)
    test_set = preprocessing_test_dataset(pd_dataset)
    return test_set


def load_data(dataset_dir: str):
    """
    Load original dataset from train.csv

    :param dataset_dir (str): a path of train.csv
    :return:
    """
    pd_dataset = pd.read_csv(dataset_dir)
    prepocessed_dataset = preprocessing_dataset(pd_dataset)
    return prepocessed_dataset


def tokenized_dataset(dataset, tokenizer):
    """tokenizer에 따라 sentence를 tokenizing 합니다."""
    concat_entity = []
    for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
        temp = ""
        temp = "[ENT]" + e01 + "[/ENT]" + "[SEP]" + "[ENT]" + e02 + "[/ENT]"

        # 주어 + 목적어 pair
        concat_entity.append(temp)
    # tokenizer => 위키피디아 한글 데이터로 만든 워드피스 토크나이저 활용
    # tokenizer = BertTokenizer(
    #     vocab_file='my_tokenizer-vocab.txt',
    #     max_len=128,
    #     do_lower_case=False,
    # )

    # 엔티티 구분용인 [SEP] 토큰까지 wordpiece 되는 현상 방지
    tokenizer.add_special_tokens({'sep_token': '[SEP]'})

    # 엔티티 강조
    tokenizer.add_special_tokens({'additional_special_tokens': ['[ENT]', '[/ENT]']})

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

    # keep entity oen hot 확인
    entity_ids = get_entity_token_ids(tokenized_sentences["input_ids"])

    # 0 or 1
    embedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=768)
    embedded_entity = embedding(entity_ids)
    print('--- Embedded Entity Ids ---')
    print(embedded_entity)
    print(embedded_entity.shape)

    embedding1 = torch.nn.Embedding(num_embeddings=32002, embedding_dim=768)
    embedded_input = embedding1(tokenized_sentences["input_ids"])
    print('--- Embedded Input Ids ---')
    print(embedded_input)
    print(embedded_input.shape)

    # broadcasting summation
    inputs_embeds = embedded_input + embedded_entity
    print(inputs_embeds)
    print(inputs_embeds.shape)

    tokenized_sentences['inputs_embeds'] = inputs_embeds
    del tokenized_sentences['input_ids']

    # tokenized_sentences['position_ids'] = entity_ids
    print("--- Print Tokenized Sentences ---")
    print(tokenized_sentences)

    print("--- Encode Tokenized Sentences ---")
    # print(tokenizer.convert_ids_to_tokens(tokenized_sentences["input_ids"][0]))

    print("--- Decode Tokenized Sentences ---")
    # print(tokenizer.decode(tokenized_sentences["input_ids"][0]))

    ################# 원복해야 함
    return tokenized_sentences, tokenizer


def make_stratifiedkfold(
    raw_train: DataFrame,
    raw_train_classes: list,
    n_splits: int,
    shuffle: bool,
    seed: int,
):
    """#### raw_train dataset을 stratifiedkfold로 나눠 주는 함수
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
        n_splits=n_splits, shuffle=shuffle, random_state=seed
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


# class CustomTrainer(Trainer):
#     """[summary]
#     Trainer에 sampler 추가
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
#         if not isinstance(self.train_dataset, collections.abc.Sized):
#             return None

#         generator = None
#         if self.args.world_size <= 1 and _is_torch_generator_available:
#             generator = torch.Generator()
#             generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

#         # Build the sampler.
#         if self.args.group_by_length:
#             if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
#                 lengths = (
#                     self.train_dataset[self.args.length_column_name]
#                     if self.args.length_column_name in self.train_dataset.column_names
#                     else None
#                 )
#             else:
#                 lengths = None
#             model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
#             if self.args.world_size <= 1:
#                 return LengthGroupedSampler(
#                     self.train_dataset,
#                     self.args.train_batch_size,
#                     lengths=lengths,
#                     model_input_name=model_input_name,
#                     generator=generator,
#                 )
#             else:
#                 return DistributedLengthGroupedSampler(
#                     self.train_dataset,
#                     self.args.train_batch_size,
#                     num_replicas=self.args.world_size,
#                     rank=self.args.process_index,
#                     lengths=lengths,
#                     model_input_name=model_input_name,
#                     seed=self.args.seed,
#                 )

#         else:
#             if self.args.world_size <= 1:
#                 if _is_torch_generator_available:
#                     return RandomSampler(self.train_dataset, generator=generator)
#                 return RandomSampler(self.train_dataset)
#             elif (
#                 self.args.parallel_mode in [ParallelMode.TPU, ParallelMode.SAGEMAKER_MODEL_PARALLEL]
#                 and not self.args.dataloader_drop_last
#             ):
#                 # Use a loop for TPUs when drop_last is False to have all batches have the same size.
#                 return DistributedSamplerWithLoop(
#                     self.train_dataset,
#                     batch_size=self.args.per_device_train_batch_size,
#                     num_replicas=self.args.world_size,
#                     rank=self.args.process_index,
#                     seed=self.args.seed,
#                 )
#             else:
#                 return DistributedSampler(
#                     self.train_dataset,
#                     num_replicas=self.args.world_size,
#                     rank=self.args.process_index,
#                     seed=self.args.seed,
#                 )
