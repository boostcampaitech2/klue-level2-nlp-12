import pickle as pickle
import os
import pandas as pd
import torch

from sklearn.model_selection import train_test_split


class RE_Dataset(torch.utils.data.Dataset):
    """
        set Dataset

    Args:
        pair_dataset (:obj: transformers.tokenization_utils_base.BatchEncoding):
            tokenized dataset
            
        labels (:obj: 1d array-like):
            int type labels(index).
    """

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
    """
        Change the first test dataset csv file you have called to the desired form of DataFrame.
        
     Args:
        dataset (:obj: pd.DataFrame):
            raw csv file of pd.DataFrame form

    Returns:
        test_dataset(:obj: pd.DataFrame):
            changed test dataset
    """
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

    test_dataset = pd.DataFrame(
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

    return test_dataset


def load_test_data(dataset_dir):
    """
        Calls the test dataset csv file to take over to the path.
    
    Args:
        dataset_dir (str):
            path of test dataset directory

    Returns:
        test_dataset(:obj: pd.DataFrame):
            changed test dataset
    """
    pd_dataset = pd.read_csv(dataset_dir)
    test_dataset = preprocessing_test_dataset(pd_dataset)

    return test_dataset


def preprocessing_train_dataset(dataset):
    """
        Change the first train dataset csv file you have called to the desired form of DataFrame.
        
     Args:
        dataset (:obj: pd.DataFrame):
            raw csv file of pd.DataFrame form

    Returns:
        train_dataset(:obj: pd.DataFrame):
            changed train dataset
        
        dev_dataset(:obj: pd.DataFrame):
            changed dev dataset
    """
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
    out_dataset = out_dataset.drop_duplicates(
        ["sentence", "subject_entity", "object_entity"], keep="first"
    )

    train_dataset, dev_dataset = train_test_split(
        out_dataset, test_size=0.2, stratify=out_dataset["label"], random_state=42
    )

    return train_dataset, dev_dataset


def load_train_data(dataset_dir):
    """
        Calls the train dataset csv file to take over to the path.
    
    Args:
        dataset_dir (str):
            path of train dataset directory

    Returns:
        train_dataset(:obj: pd.DataFrame):
            changed train dataset
        
        dev_dataset(:obj: pd.DataFrame):
            changed dev dataset

    """
    pd_dataset = pd.read_csv(dataset_dir)
    train_dataset, dev_dataset = preprocessing_train_dataset(pd_dataset)

    return train_dataset, dev_dataset


def preprocessing_dataset(dataset):
    """
        Change the first dataset csv file you have called to the desired form of DataFrame.
        
     Args:
        dataset (:obj: pd.DataFrame):
            raw csv file of pd.DataFrame form

    Returns:
        out_dataset(:obj: pd.DataFrame):
            changed dataset
    """
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
    out_dataset = out_dataset.drop_duplicates(
        ["sentence", "subject_entity", "object_entity"], keep="first"
    )

    return out_dataset


def load_data(dataset_dir):
    """
        Calls the dataset csv file to take over to the path.
    
    Args:
        dataset_dir (str):
            path of dataset directory

    Returns:
        out_dataset(:obj: pd.DataFrame):
            changed dataset
    """
    pd_dataset = pd.read_csv(dataset_dir)
    out_dataset = preprocessing_dataset(pd_dataset)

    return out_dataset


# https://huggingface.co/transformers/internal/tokenization_utils.html
def tokenized_dataset(dataset, tokenizer, token_type, model_name):
    """
        According to tokenizer, the sentence is tokenizing.
    
    Args:
        dataset(:obj: pd.DataFrame):
            tokenizing dataset
        
        tokenizer(:obj: tokenizer):
            Tokenizer to fit model
        
        token_type(str):
            set token_type
            options : 'default', 'swap_entity', 'sentence_entity', 'punct_typed_entity'
        
        model_name(str):
            set model_name.
            e.g. 'klue/roberta-large', 'xlm-roberta-large', 'klue/bert-base'

    Returns:
        tokenized_sentences(:obj: transformers.tokenization_utils_base.BatchEncoding):
            tokenized dataset
    """
    concat_entity = []
    if 'xlm' in model_name:
        if token_type == "default":
            for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
                temp = ""
                temp = e01 + "</s>" + e02
                concat_entity.append(temp)
        if token_type == "swap_entity":
            for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
                temp = ""
                temp = e02 + "</s>" + e01
                concat_entity.append(temp)
        elif token_type == "sentence_entity":
            for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
                temp = ""
                temp = f"{e01}과 {e02}는 어떤 관계일까?"  # f-string
                concat_entity.append(temp)
        elif token_type == "punct_typed_entity":
            for e01, e02, t01, t02 in zip(
                dataset["subject_entity"],
                dataset["object_entity"],
                dataset["subject_entity_type"],
                dataset["object_entity_type"],
            ):
                temp = ""
                temp = f"@ * {t01} * {e01} @ # ^ {t02} ^ {e02} #"
                concat_entity.append(temp)
    else:
        if token_type == "default":
            for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
                temp = ""
                temp = e01 + "[SEP]" + e02
                concat_entity.append(temp)
        if token_type == "swap_entity":
            for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
                temp = ""
                temp = e02 + "[SEP]" + e01
                concat_entity.append(temp)
        elif token_type == "sentence_entity":
            for e01, e02 in zip(dataset["subject_entity"], dataset["object_entity"]):
                temp = ""
                temp = f"{e01}과 {e02}는 어떤 관계일까?"  # f-string
                concat_entity.append(temp)
        elif token_type == "punct_typed_entity":
            for e01, e02, t01, t02 in zip(
                dataset["subject_entity"],
                dataset["object_entity"],
                dataset["subject_entity_type"],
                dataset["object_entity_type"],
            ):
                temp = ""
                temp = f"@ * {t01} * {e01} @ # ^ {t02} ^ {e02} #"
                concat_entity.append(temp)

    tokenized_sentences = None

    if model_name == 'klue/bert-base':
        tokenized_sentences = tokenizer(
            concat_entity,
            list(dataset["sentence"]),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
            add_special_tokens=True,
            # return_token_type_ids=False,
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
