import pickle as pickle
import os
import pandas as pd
import sklearn
import random
import wandb
import argparse
import numpy as np
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import class_weight
from torchsampler import ImbalancedDatasetSampler # https://github.com/ufoym/imbalanced-dataset-sampler
from torch.utils.data import DataLoader, Dataset, IterableDataset, RandomSampler, SequentialSampler, WeightedRandomSampler
from torchsummary import summary
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import (
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
    TrainerCallback,
)
from transformers.integrations import WandbCallback # https://huggingface.co/transformers/main_classes/callback.html?highlight=callbacks
from transformers.file_utils import is_datasets_available, is_sagemaker_mp_enabled
from torch.cuda.amp import autocast
from adamp import AdamP
#from apex import amp
from load_data import *


LABEL_LIST = [ # in-order
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

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    no_relation_label_idx = LABEL_LIST.index("no_relation")
    label_indices = list(range(len(LABEL_LIST)))
    label_indices.remove(no_relation_label_idx)
    return sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices) * 100.0

def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(targets_c, preds_c)
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0

def compute_metrics(pred):
  """ validation을 위한 metrics function """
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  probs = pred.predictions

  # calculate accuracy using sklearn's function
  f1 = klue_re_micro_f1(preds, labels)
  auprc = klue_re_auprc(probs, labels)
  acc = accuracy_score(labels, preds) # 리더보드 평가에는 포함되지 않습니다.

  return {
      'micro f1 score': f1,
      'auprc' : auprc,
      'accuracy': acc,
  }

def label_to_num(label):
  num_label = []
  with open('dict_label_to_num.pkl', 'rb') as f:
    dict_label_to_num = pickle.load(f)
  for v in label:
    num_label.append(dict_label_to_num[v])
  
  return num_label


class FocalLoss(nn.Module): #V2
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')
        loss = ce_loss(inputs, targets)

        pt = torch.exp(-loss)
        F_loss = self.alpha * (1-pt)**self.gamma * loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class CustomTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]):# -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.
        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model.train()

        inputs = self._prepare_inputs(inputs)
        labels = inputs.pop('labels')

        criterion = FocalLoss()

        outputs = model(**inputs)
        if self.use_amp:
            with autocast():
                loss = criterion(outputs['logits'], labels)
        else:
            loss = criterion(outputs['logits'], labels)


        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()


def train(args):
    # load model and tokenizer
    #MODEL_NAME = "klue/bert-base"
    #MODEL_NAME = "klue/roberta-base"
    #MODEL_NAME = "xlm-roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # load dataset
    train_dataset = load_data("../dataset/train/train_jap_swap_final.csv")
    train_label = label_to_num(train_dataset["label"].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer, model='roberta', token_type='punct_typed_entity')

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(args.model)
    model_config.num_labels = 30

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, config=model_config
    )

    params = model.parameters()
    optimizer = AdamP(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    epoch_steps = len(train_label) // args.train_batch_size
    t_max = (epoch_steps * args.epochs)
    print('=========================')
    print(f'T_max : {t_max}')
    print('=========================')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-6)
    
    #model.parameters
    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        seed=args.seed,
        output_dir=args.output_dir,  # output directory
        save_total_limit=args.save_total_limit,  # number of total save model.
        save_steps=args.save_steps,  # model saving step.
        num_train_epochs=args.epochs,  # total number of training epochs
        learning_rate=args.lr,  # learning_rate #5e-5
        per_device_train_batch_size=args.train_batch_size,  # batch size per device during training 64
        per_device_eval_batch_size=args.eval_batch_size,  # batch size for evaluation 64
        warmup_steps=args.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=args.weight_decay,  # strength of weight decay
        logging_dir=args.logging_dir,  # directory for storing logs
        logging_steps=args.logging_steps,  # log saving step.
        evaluation_strategy=args.evaluation_strategy,  # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=args.eval_steps,  # evaluation step.
        load_best_model_at_end=True,
        report_to='wandb'
    )

    trainer = CustomTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_train_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],#, TrainCallback],
        optimizers=(optimizer, scheduler)
    )

    # train model
    trainer.train()
    model.save_pretrained(args.best_model_dir)

def main(args):
    seed_everything(args.seed)
    train(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # train args
    parser.add_argument("--seed", type=int, default=42, help="seed value (default: 42)")
    parser.add_argument("--model", type=str, default="xlm-roberta-large", help="model type (default: klue/roberta-large)")
    parser.add_argument("--output_dir", type=str, default="./results", help="output directory (default: ./results)")
    parser.add_argument("--save_total_limit", type=int, default=3, help="number of total save model (default: 3)")
    parser.add_argument("--save_steps", type=int, default=906, help="model saving step (default: 500)")
    parser.add_argument("--epochs", type=int, default=4, help="number of epochs to train (default: 5)")
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate (default: 5e-5)")
    parser.add_argument("--train_batch_size", type=int, default=64, help="train batch size (default: 64)")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="eval batch size (default: 64)")
    parser.add_argument("--warmup_steps", type=int, default=500, help="lambda lr scheduler warmup steps (default: 500)")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="adam optimizer weight decay (default: 1e-2)")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="log directory (default: ./logs)")
    parser.add_argument("--logging_steps", type=int, default=906, help="log saving step (default: 100)")
    parser.add_argument("--evaluation_strategy", type=str, default="steps",
        help="""evaluation strategy to adopt during training (default: step)
                `no`: No evaluation during training.
                `steps`: Evaluate every `eval_steps`.
                `epoch`: Evaluate every end of epoch.""",
    )
    parser.add_argument("--eval_steps", type=int, default=906, help="evaluation step (default: 500)")
    parser.add_argument("--best_model_dir", type=str, default="./best_model/[xlm-roberta-large] punct_typed_entity", help="best model direcotry(default: ./best_model")
    
    parser.add_argument('--wandb_project', type=str, default='train_jap_swap_final', help='wandb project name (default: klue-re')

    args = parser.parse_args()
    # 1. Start a new run
    #os.environ['WANDB_WATCH'] = 'all'
    wandb.init(project=args.wandb_project, entity='tttangmin', name='xlm-roberta-large-punct')

    main(args)

    wandb.finish()