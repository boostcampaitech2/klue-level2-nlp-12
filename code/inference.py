from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import torch.nn.functional as F

import pickle as pickle
import numpy as np
import argparse
from tqdm import tqdm

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
  dataloader = DataLoader(tokenized_sent, batch_size=16, shuffle=False)
  model.eval()
  output_pred = []
  output_prob = []
  for i, data in enumerate(tqdm(dataloader)):
    with torch.no_grad():
      if model_name == 'klue/bert-base':
        outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
        )
      else:
         outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          #token_type_ids=data['token_type_ids'].to(device)
        )   
    logits = outputs[0]
    prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    output_pred.append(result)
    output_prob.append(prob)
  
  return np.concatenate(output_pred).tolist(), np.concatenate(output_prob, axis=0).tolist()

def num_to_label(label):
  """
  Converts a class from a number to an original string label.

  Args:
    label (1d array-like): Labels converted to numbers

  Returns:
    origin_label (list): origin label (string)
  """
  origin_label = []

  # open pickle file
  with open('dict_num_to_label.pkl', 'rb') as f:
    dict_num_to_label = pickle.load(f)
  for v in label:
    origin_label.append(dict_num_to_label[v])
  
  return origin_label

def load_test_dataset(dataset_dir, tokenizer, token_type, model_name):
  """
  Tokenizing after loading the test dataset.

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
  test_dataset = load_data(dataset_dir)
  test_label = list(map(int,test_dataset['label'].values))

  # tokenizing dataset
  # token_type options : 'default', 'swap_entity', 'sentence_entity', 'punct_typed_entity'
  # model_name options : 'klue/roberta-large', 'xlm-roberta-large', 'klue/bert-base'
  tokenized_test = tokenized_dataset(test_dataset, tokenizer, token_type, model_name)

  return test_dataset['id'], tokenized_test, test_label

def main(args):
  """
  If the given dataset is in the same format as the csv file, inference is possible.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  tokenizer = AutoTokenizer.from_pretrained(args.model)

  ## load my best model
  model = AutoModelForSequenceClassification.from_pretrained(args.best_model_dir)
  model.parameters
  model.to(device)

  ## load test datset
  test_dataset_dir = "../dataset/test/test_data.csv"
  test_id, test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer, args.token_type, args.model)
  Re_test_dataset = RE_Dataset(test_dataset, test_label)

  ## predict answer
  pred_answer, output_prob = inference(model, Re_test_dataset, device, args.model) # model에서 class 추론
  pred_answer = num_to_label(pred_answer) # 숫자로 된 class를 원래 문자열 라벨로 변환.
  
  ## make csv file with predicted answer
  #########################################################
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
  output = pd.DataFrame({'id':test_id,'pred_label':pred_answer,'probs':output_prob,})

  output.to_csv('./prediction/submission.csv', index=False) # 최종적으로 완성된 예측한 라벨 csv 파일 형태로 저장.
  #### 필수!! ##############################################
  print('---- Finish! ----')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  parser.add_argument("--best_model_dir", type=str, default="./best_model", help="best model direcotry(default: ./best_model")
  parser.add_argument("--model", type=str, default="klue/roberta-large", help="model type (default: klue/roberta-large)")
  parser.add_argument('--token_type', type=str, default='default', help='entity token marker type (default: default)')

  args = parser.parse_args()
  print(args)

  main(args)
  
