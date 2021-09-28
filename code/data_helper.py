from tkinter.tix import Tree
from load_data import *
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import os
"""
inference 시 train dataset에서 stratify 하게 가져와 제출 전 training 되지 않은 데이터로 metric 계산을 할 때 도와줌
"""

TRAIN_FILE_PATH = '/opt/ml/dataset/train/train.csv'
TRAIN_FOLDER_PATH = '/opt/ml/dataset/train'
FOLD_NUM = 10
# ROLLBACK = True
ROLLBACK = False
SEED = 42

raw_train_df = pd.read_csv(TRAIN_FILE_PATH)


def change_origin_file_name(file_path, new_name: str = 'old_train.csv'):
    NEW_TRAIN_FILE_PATH = os.path.join(os.path.split(file_path)[0], new_name)

    os.rename(file_path, NEW_TRAIN_FILE_PATH)


def back2raw():
    """
    다시 원래 데이터로 돌려주는 함수
    """
    file_lists = os.listdir(TRAIN_FOLDER_PATH)
    if 'old_train.csv' in file_lists:
        for file in file_lists:
            if file != 'old_train.csv' and os.path.isfile(os.path.join(TRAIN_FOLDER_PATH, file)):
                print(file)
                os.remove(os.path.join(TRAIN_FOLDER_PATH, file))
        os.rename(os.path.join(TRAIN_FOLDER_PATH,
                  'old_train.csv'), TRAIN_FILE_PATH)


if __name__ == '__main__':
    if ROLLBACK:
        back2raw()
    else:
        folds = StratifiedKFold(
            n_splits=FOLD_NUM,
            shuffle=True,
            random_state=SEED
        ).split(np.arange(raw_train_df.shape[0]), raw_train_df['label'])

        change_origin_file_name(TRAIN_FILE_PATH)
        for fold, (train_idx, test_idx) in enumerate(folds):
            if fold > 0:
                break
            train_ = raw_train_df.loc[train_idx, :].reset_index(drop=True)
            train_.to_csv(os.path.join(
                TRAIN_FOLDER_PATH, 'train.csv'), index=False)

            test_ = raw_train_df.loc[test_idx, :].reset_index(drop=True)
            test_.to_csv(os.path.join(TRAIN_FOLDER_PATH,
                         'custom_test.csv'), index=False)
