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

### Hyperparameter Search
```
+------------------------+------------+-------+-----------------+--------------------+-------------------------------+-------------+
| Trial name             | status     | loc   |   learning_rate |   num_train_epochs |   per_device_train_batch_size |   objective |
|------------------------+------------+-------+-----------------+--------------------+-------------------------------+-------------|
| _objective_6b574_00000 | TERMINATED |       |     2.80576e-05 |                  5 |                           128 |     154.363 |
| _objective_6b574_00001 | TERMINATED |       |     0.000145532 |                  5 |                            32 |     149.825 |
| _objective_6b574_00002 | TERMINATED |       |     1.02567e-05 |                  5 |                           128 |     132.344 |
| _objective_6b574_00003 | TERMINATED |       |     6.53337e-06 |                  6 |                            32 |     147.45  |
| _objective_6b574_00004 | TERMINATED |       |     7.96526e-05 |                  6 |                           128 |     160.306 |
| _objective_6b574_00005 | TERMINATED |       |     5.49717e-06 |                  6 |                           256 |     106.767 |
| _objective_6b574_00006 | TERMINATED |       |     0.000231129 |                  6 |                            64 |     143.499 |
| _objective_6b574_00007 | TERMINATED |       |     1.1551e-05  |                  5 |                            32 |     154.669 |
| _objective_6b574_00008 | TERMINATED |       |     2.02981e-05 |                  6 |                            32 |     160.616 |
| _objective_6b574_00009 | TERMINATED |       |     3.65477e-05 |                  5 |                           128 |     157.183 |
+------------------------+------------+-------+-----------------+--------------------+-------------------------------+-------------+

BestRun(run_id='6b574_00008', objective=160.6155945027986, hyperparameters={'learning_rate': 2.0298058052421517e-05, 'num_train_epochs': 6, 'per_device_train_batch_size': 32})
```

### Wandb for Tracking
![wandb_img](https://user-images.githubusercontent.com/34739974/136650025-d92d7fac-5967-4cb7-8469-55a74f309e18.PNG)
