# Team AI-it Repo for KLUE tasks

## Contributor
- 김재현/T2050
- 박진영/T2096
- 안성민/T2127
- 양재욱/T2130
- 이연걸/T2163
- 조범준/T2211
- 진혜원/T2217

## Experiment Log
- [Experiment Log](https://jet-rook-fae.notion.site/NLP-KLUE-Experiment-Log-b0ee85a289404de9852c579ef7d9b5e5)


## Hardware
- GPU : V100
- Language : Python
- Develop tools : Jupyter Notebook, VSCode, Pycharm, Google Colab


## File List
```
code/
  ├── train
  ├── inference
  ├── inference_ensemble
  ├── data_helper
  ├── load_data
  ├── utils
  ├── error_handler
  └── requirements.txt
dataset/
  └── train
      └── train.csv
  └── test
      └── test.csv
```

## Getting Started
### Dependencies
- torch==1.6.0
- transformers==4.11.0
- wandb==0.12.3
- koeda==0.0.4
- konlpy==0.5.2
- kss==3.2.0
- soynlp==0.0.493

### Install Requirements
```
pip install -r requirements.txt
```

### Training
```
python train.py --[args] [value]
python train.py --model klue-roberta-large
```

### Inference
- TTA(Test Time Augmentation) 사용
```
python inference.py --[args] [value]
python inference.py --ensemble True

python inference_ensemble.py --[args] [value]
python inference_ensemble.py --model_dir ./ensemble
```

## Apply
### Dataset
- Train : Given KLUE dataset

### Model
- `klue/roberta-large` `klue/bert-base`

### Optimizer & Loss
- Optimizer : AdamP
- Loss : Focal Loss

### LR Scheduler
- CosineAnnealingLR
