from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    RagSequenceForGeneration,
)
from torch.utils.data import DataLoader
from load_data import *
from utils import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import classification_report
import os
from train import label_to_num
from .error_handler import *
import re


"""
ensemble 폴더에 앙상블 할 각 모델이 담긴 폴더의 형식을 
[모델명] token_type
로 통일 해주시면 됩니다.
"""


def inference(model, tokenized_sent, device, model_name):
    """
    After making the test dataset as a DataLoader,
    the model predicts it by dividing it by batch_size.

    Args:
        model (:obj: `nn.Module`):
            Load the trained model
        tokenized_sent (:obj: `torch.utils.data.Dataset`):
            Load the tokenized statement
        device: (:obj: `torch.device`):
            CUDA or CPU
        model_name (str):
            Option - klue/roberta-large, klue/bert-base, xlm-roberta-large

    Return:
        output_pred (list), output_prob (list) : tuple of lists
            output_pred : Concat of the labels predicted by the model
            output_prob : Concat of the probabilities for the labels predicted by the model
    """

    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False, num_workers=4)
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            if model_name == 'klue/bert-base':
                outputs = model(
                    input_ids=data["input_ids"].to(device),
                    attention_mask=data["attention_mask"].to(device),
                    token_type_ids=data['token_type_ids'].to(device) # klue/roberta 관련 모델의 경우, 주석 처리
                )
            else:
                outputs = model(
                    input_ids=data["input_ids"].to(device),
                    attention_mask=data["attention_mask"].to(device),
                    # token_type_ids=data['token_type_ids'].to(device) # klue/roberta 관련 모델의 경우, 주석 처리
                )
        logits = outputs[0]
        prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
        logits = logits.detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        output_pred.append(result)
        output_prob.append(prob)

    return (
        np.concatenate(output_pred).tolist(),
        np.concatenate(output_prob, axis=0).tolist(),
    )


def num_to_label(label):
    """Converts a class from a number to an original string label.

    Args:
        label (1d array-like): Labels converted to numbers

    Returns:
        origin_label (list): origin label (string)
    """
    origin_label = []

    # open pickle file
    with open("dict_num_to_label.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label


def load_test_dataset(
    dataset_dir, tokenizer, token_type="default", model_name="klue/roberta-large"
):
    """Tokenizing after loading the test dataset.

    Args:
        dataset_dir (str): Dataset directory path
        tokenizer : The right tokenizer for your model
        token_type (str): Option - default, swap_entity, sentence_entity, punct_typed_entity
        model_name (str): Option - klue/roberta-large, klue/bert-base, xlm-roberta-large

    Returns:
        test_dataset["id"], tokenized_test, test_label: Tuple of arrays
            test_dataset["id"] : IDs of test dataset
            tokenized_test : tokenized test sentance
            test_label : Labels of test dataset
    """
    try:
        if token_type not in [
            "default",
            "swap_entity",
            "sentence_entity",
            "punct_typed_entity",
        ]:
            print(
                f"{token_type} is not in [default, swap_entity, sentence_entity, punct_typed_entity]"
            )
            print("Please check saved folder name")
            print("Folder name : [{backbone model name}] {entity_option}")
            print(
                "Warning !! : If you use the klue model, change / to - and set it as the folder name."
            )
            raise CustomError(ErrorCode.NOT_FOUND, ErrorMsg.WRONG_INPUT)

        if model_name not in [
            "klue/roberta-large",
            "klue/bert-base",
            "xlm-roberta-large"
        ]:
            print(
                f"{model_name} is not in [klue/roberta-large, klue/bert-base, xlm-roberta-large]"
            )
            raise CustomError(ErrorCode.BAD_REQUEST, ErrorMsg.WRONG_INPUT)

        test_dataset = load_data(dataset_dir)
        test_label = list(map(int, test_dataset["label"].values))
        tokenized_test = tokenized_dataset(
            test_dataset, tokenizer, token_type, model_name
        )
        return test_dataset["id"], tokenized_test, test_label
    except CustomError as e:
        print(e)


def main(args):
    """
    If the given dataset is in the same format as the csv file, inference is possible.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load test datset
    test_dataset_dir = "/opt/ml/dataset/test/test_data.csv"

    # Just put the data to assemble into the ensemble folder as a folder.
    output_probs = []
    ensemble_dir = "./ensemble"
    try:
        if not os.path.exists(ensemble_dir):
            print("The ensemble folder does not exist in the same folder as that file")
            raise CustomError(ErrorCode.NOT_FOUND, ErrorMsg.WRONG_INPUT)
    except:
        for folder in os.listdir(ensemble_dir):
            print(folder)
            arc_name = folder.split(" ")[0][1:-1]

            if "klue" in arc_name:
                arc_name = (
                    arc_name.split("-")[0] + "/" + "-".join(arc_name.split("-")[1:])
                )
            
            tokenizer = AutoTokenizer.from_pretrained(arc_name)

            test_id, test_dataset, test_label = load_test_dataset(
                test_dataset_dir, tokenizer, folder.split(" ")[-1], arc_name
            )

            Re_test_dataset = RE_Dataset(test_dataset, test_label)

            model = AutoModelForSequenceClassification.from_pretrained(
                os.path.join(ensemble_dir, folder)
            )
            model.parameters
            model.to(device)

            # predict answer
            pred_answer, output_prob = inference(
                model, Re_test_dataset, device, arc_name
            )  # Infer class from model
            output_probs.append(np.array(output_prob))

        if args.ensemble_option == "mean":
            mean_prob = np.mean(output_probs, axis=0)
            pred_answer = np.argmax(mean_prob, axis=-1)

    # custom predict answer
    pred_answer = num_to_label(
        pred_answer
    )  # Convert numeric classes to original string labels.

    # make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame(
        {"id": test_id, "pred_label": pred_answer, "probs": output_prob}
    )

    # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    output.to_csv("./prediction/submission_ensemble.csv", index=False)
    #### 필수!! ##############################################

    print("---- Finish! ----")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model dir
    parser.add_argument(
        "--ensemble_dir",
        type=str,
        default="./ensemble",
        help="path of ensemble folder (defualt : ./ensemble)",
    )
    parser.add_argument(
        "--ensemble_option",
        type=str,
        default="mean",
        help="ensemble method (default : mean)",
    )

    args = parser.parse_args()
    print(args)
    
    main(args)
