from sklearn import ensemble
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import DataLoader
from load_data import *
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


def inference(model, tokenized_sent, device):
    """
    test dataset을 DataLoader로 만들어 준 후,
    batch_size로 나눠 model이 예측 합니다.
    """
    dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False, num_workers=4)
    model.eval()
    output_pred = []
    output_prob = []
    for i, data in enumerate(tqdm(dataloader)):
        with torch.no_grad():
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
    """
    숫자로 되어 있던 class를 원본 문자열 라벨로 변환 합니다.
    """
    origin_label = []

    # open pickle file
    with open("dict_num_to_label.pkl", "rb") as f:
        dict_num_to_label = pickle.load(f)
    for v in label:
        origin_label.append(dict_num_to_label[v])

    return origin_label


def load_test_dataset(dataset_dir, tokenizer):
    """
    test dataset을 불러온 후,
    tokenizing 합니다.
    """
    test_dataset = load_test_data(dataset_dir)
    test_label = list(map(int, test_dataset["label"].values))

    #### 원복해야 함
    tokenized_test, tokenizer = tokenized_dataset(test_dataset, tokenizer)
    return test_dataset["id"], tokenized_test, test_label


def custom_load_test_dataset(dataset_dir, tokenizer):
    """
    test dataset을 불러온 후,
    tokenizing 합니다.
    """
    test_dataset = load_test_data(dataset_dir)
    test_label = list(map(int, label_to_num(test_dataset["label"].values)))


    #### 원복해야 함
    tokenized_test, tokenizer = tokenized_dataset(test_dataset, tokenizer)
    return test_dataset["id"], tokenized_test, test_label


def main(args):
    """
    주어진 dataset csv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load tokenizer
    # Tokenizer_NAME = "klue/bert-base"
    Tokenizer_NAME = "klue/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(Tokenizer_NAME)

    # load my model
    MODEL_NAME = args.model_dir  # model dir.

    ############# keep 임시
    args.model_dir = './best_model/2021-09-30 14:46:57/klue-roberta-small-Fold0'
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    print(model.parameters)

    # load test datset
    test_dataset_dir = "/opt/ml/dataset/test/test_data.csv"
    test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
    Re_test_dataset = RE_Dataset(test_dataset, test_label)

    # load custom test dataset
    custom_test_dataset_dir = "/opt/ml/dataset/train/custom_test.csv"
    custom_test_id, custom_test_dataset, custom_test_label = custom_load_test_dataset(
        custom_test_dataset_dir, tokenizer
    )
    custom_Re_test_dataset = RE_Dataset(custom_test_dataset, custom_test_label)

    if args.ensemble:
        # ensemble 폴더에 어셈블할 데이터를 폴더채로 넣으시면 됩니다
        output_probs = []
        custom_output_probs = []
        ensemble_dir = "./ensemble"

        ensemble_dir_list = [
            './best_model/klue-roberta-large-Fold0',
            './best_model/klue-roberta-large-Fold1',
            './best_model/klue-roberta-large-Fold2',
            './best_model/klue-roberta-large-Fold3'
        ]


        # for folder in os.listdir(ensemble_dir):
        for folder in ensemble_dir_list:
            model = AutoModelForSequenceClassification.from_pretrained(
                # os.path.join(ensemble_dir, folder)
                folder
            )
            print(model.parameters)
            model.to(device)
            # predict answer
            pred_answer, output_prob = inference(
                model, Re_test_dataset, device
            )  # model에서 class 추론
            custom_pred_answer, custom_output_prob = inference(
                model, custom_Re_test_dataset, device
            )  # model에서 class 추론

            output_probs.append(np.array(output_prob))
            custom_output_probs.append(np.array(custom_output_prob))

        if args.ensemble_option == "mean":
            mean_prob = np.mean(output_probs, axis=0)
            custom_mean_prob = np.mean(custom_output_probs, axis=0)
            pred_answer = np.argmax(mean_prob, axis=-1)
            custom_pred_answer = np.argmax(custom_mean_prob, axis=-1)

        # custom predict answer
        # 숫자로 된 class를 원래 문자열 라벨로 변환.
        pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환.
        # custom_pred_answer = num_to_label(custom_pred_answer)

    else:
        # predict answer
        pred_answer, output_prob = inference(
            model, Re_test_dataset, device
        )  # model에서 class 추론
        pred_answer = num_to_label(pred_answer)  # 숫자로 된 class를 원래 문자열 라벨로 변환.

        # custom predict answer
        custom_pred_answer, custom_output_prob = inference(
            model, custom_Re_test_dataset, device
        )  # model에서 class 추론
        # 숫자로 된 class를 원래 문자열 라벨로 변환.
        # custom_pred_answer = num_to_label(custom_pred_answer)

    # make csv file with predicted answer
    #########################################################
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame(
        {"id": test_id, "pred_label": pred_answer, "probs": output_prob}
    )

    # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
    output.to_csv("./prediction/submission.csv", index=False)
    #### 필수!! ##############################################
    print(classification_report(custom_test_label, custom_pred_answer))
    print("---- Finish! ----")


if __name__ == "__main__":
    # disable warning log
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()

    # model dir
    parser.add_argument("--model_dir", type=str, default="./best_model")
    parser.add_argument("--ensemble", type=bool, default=False)
    parser.add_argument(
        "--ensemble_option", type=str, default="mean", help="mean, hard_voting"
    )
    args = parser.parse_args()
    print(args)
    main(args)
