import pickle as pickle
import os
from pyexpat import model
import pandas as pd
import torch
import sklearn
import numpy as np
import random
import argparse

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)
from transformers import (
    XLMRobertaForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    RobertaConfig,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    BertTokenizer,
    EarlyStoppingCallback,
)

from load_data import *
from utils import *

TIME = korea_now()


def seed_everything(seed: int = 42):
    """
    Fix all related seeds

    :param seed: 42 (default)
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = [
        "no_relation",
        "org:top_members/employees",
        "org:members",
        "org:product",
        "per:title",
        "org:alternate_names",
        "per:employee_of",
        "org:place_of_headquarters",
        "per:product",
        "org:number_of_employees/members",
        "per:children",
        "per:place_of_residence",
        "per:alternate_names",
        "per:other_family",
        "per:colleagues",
        "per:origin",
        "per:siblings",
        "per:spouse",
        "org:founded",
        "org:political/religious_affiliation",
        "org:member_of",
        "per:parents",
        "org:dissolved",
        "per:schools_attended",
        "per:date_of_death",
        "per:date_of_birth",
        "per:place_of_birth",
        "per:place_of_death",
        "org:founded_by",
        "per:religion",
    ]
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return (
        sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices)
        * 100.0
    )


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(
            targets_c, preds_c
        )
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def compute_metrics(pred):
    """validation을 위한 metrics function"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

    return {
        "micro f1 score": f1,
        "auprc": auprc,
        "accuracy": acc,
    }


def label_to_num(label):
    num_label = []
    with open("dict_label_to_num.pkl", "rb") as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])
    return num_label


def train(args):
    # load model and tokenizer
    # MODEL_NAME = "bert-base-uncased"
    # MODEL_NAME = "klue/roberta-base"
    # MODEL_NAME = "klue/roberta-large"
    # MODEL_NAME = "xlm-roberta-large"
    # MODEL_NAME = "roberta-large"
    MODEL_NAME = args.model  # defalut : klue/roberta-base
    if args.title == None:
        TITLE = args.model.replace("/", "-")
    else:
        TITLE = args.title
    SEED = args.seed
    KFLOD_NUM = args.fold_num

    # tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    # tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    # train_dataset, dev_dataset = load_data("../dataset/train/train.csv")
    # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.
    raw_df = pd.read_csv("/opt/ml/dataset/train/train.csv")

    # set StratifiedKFold
    folds = make_stratifiedkfold(raw_df, raw_df.label, KFLOD_NUM, True, SEED)
    for fold, (trn_idx, dev_idx) in enumerate(folds):
        if not args.run_kflod:
            if fold > 0:
                break
        train_dataset, dev_dataset = make_train_df(raw_df, trn_idx, dev_idx)

        # entity extraction by eval
        train_dataset = preprocessing_dataset(train_dataset)
        dev_dataset = preprocessing_dataset(dev_dataset)

        # label(str) => label(int)
        train_label = label_to_num(train_dataset["label"].values)
        dev_label = label_to_num(dev_dataset["label"].values)

        # tokenizing dataset
        tokenized_train = tokenized_dataset(train_dataset, tokenizer)
        tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

        # make dataset for pytorch
        RE_train_dataset = RE_Dataset(tokenized_train, train_label)
        RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        # setting model hyperparameter
        model_config = AutoConfig.from_pretrained(MODEL_NAME)
        # model_config = XLMRobertaConfig.from_pretrained(MODEL_NAME)
        model_config.num_labels = 30

        # model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, config=model_config
        )
        # model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
        print(model.config)

        # exit()

        model.parameters
        model.to(device)

        # 사용한 option 외에도 다양한 option들이 있습니다.
        # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.

        training_args = TrainingArguments(
            output_dir=args.output_dir + "/" + TITLE,  # output directory
            save_total_limit=args.save_total_limit,  # number of total save model.
            save_steps=args.save_steps,  # model saving step.
            num_train_epochs=args.epochs,  # total number of training epochs
            learning_rate=args.lr,  # learning_rate
            per_device_train_batch_size=args.train_batch_size,  # batch size per device during training
            per_device_eval_batch_size=args.eval_batch_size,  # batch size for evaluation
            warmup_steps=args.warmup_steps,  # number of warmup steps for learning rate scheduler
            weight_decay=args.weight_decay,  # strength of weight decay
            logging_dir=args.logging_dir + "/" + TITLE,  # directory for storing logs
            logging_steps=args.logging_steps,  # log saving step.
            evaluation_strategy=args.evaluation_strategy,  # evaluation strategy to adopt during training
            # `no`: No evaluation during training.
            # `steps`: Evaluate every `eval_steps`.
            # `epoch`: Evaluate every end of epoch.
            eval_steps=args.eval_steps,  # evaluation step.
            load_best_model_at_end=True,
            dataloader_num_workers=4,
        )
        trainer = Trainer(
            # the instantiated 🤗 Transformers model to be trained
            model=model,
            args=training_args,  # training arguments, defined above
            train_dataset=RE_train_dataset,  # training dataset
            eval_dataset=RE_dev_dataset,  # evaluation dataset
            compute_metrics=compute_metrics,  # define metrics function
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # train model
        trainer.train()
        model.save_pretrained(
            args.save_name + TITLE + "/" + TITLE + "-Fold" + str(fold)
        )

        del model, trainer, training_args
        torch.cuda.empty_cache()


def main(args):
    seed_everything(args.seed)
    train(args)


if __name__ == "__main__":
    # disable warning log
    import os

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()

    # seed and model args
    parser.add_argument("--seed", type=int, default=42, help="seed value (default: 42)")
    parser.add_argument(
        "--model",
        type=str,
        default="klue/roberta-base",
        help="model type (default: klue/roberta-base)",
    )

    # train args
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=5,
        help="number of total save model (default: 5)",
    )
    parser.add_argument(
        "--save_steps", type=int, default=500, help="model saving step (default: 5)"
    )

    parser.add_argument(
        "--epochs", type=int, default=6, help="number of epochs to train (default: 6)"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="learning rate (default: 5e-5)"
    )

    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=64,
        help="input batch size for validing (default: 64)",
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=500,
        help="strength of weight decay (default: 200)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="number of total save model (default: 0.01)",
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default="./logs",
        help="directory for storing logs (default: ./logs)",
    )
    parser.add_argument(
        "--logging_steps", type=int, default=100, help="log saving step (default: 100)"
    )

    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="steps",
        help="""evaluation strategy to adopt during training
                `no`: No evaluation during training.
                `steps`: Evaluate every `eval_steps`.
                `epoch`: Evaluate every end of epoch.""",
    )
    parser.add_argument(
        "--eval_steps", type=int, default=500, help="evaluation step (default: 500)"
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="./best_model",
        help="model save at {save_name}",
    )
    parser.add_argument(
        "--run_kflod",
        type=bool,
        default=False,
        help="whether to use kfold(default: False)",
    )

    # directory args
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="output directory (default: ./results)",
    )
    parser.add_argument(
        "--fold_num",
        type=int,
        default=5,
        help="k in the k-fold cross validation (default: 5)",
    )
    parser.add_argument(
        "--title", type=str, default=None, help="set folder name (default: model name)"
    )
    # parser.add_argument('--tokenizer', type=str, default='steps', help='''select tokenizer
    #                                                                      klue/roberta-base: AutoTokenizer.from_pretrained(MODEL_NAME).
    #                                                                      ''')

    args = parser.parse_args()
    print("--- Args List ---")
    print(args)

    main(args)
