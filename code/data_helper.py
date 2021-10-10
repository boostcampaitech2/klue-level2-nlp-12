from tkinter.tix import Tree
from load_data import *
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import os
import argparse

"""
It is a data helper that maintains the data distribution in the train dataset during inference and divides it into the train dataset and the eval dataset.
"""


FOLD_NUM = 10
# ROLLBACK = True
ROLLBACK = False
SEED = 42


def change_origin_file_name(file_path: str, new_name: str = "old_train.csv"):
    """
    Rename the existing `train.py` file to `new_name`    
    
    Args:
        file_path (str): Original data path
        new_name (str, optional): Dataset file name for training. (Default : 'old_train.csv').
    """
    NEW_TRAIN_FILE_PATH = os.path.join(os.path.split(file_path)[0], new_name)
    os.rename(file_path, NEW_TRAIN_FILE_PATH)


def back2raw(TRAIN_FOLDER_PATH, TRAIN_FILE_PATH):
    """
    Function that returns the original data
    """
    file_lists = os.listdir(TRAIN_FOLDER_PATH)
    if "old_train.csv" in file_lists:
        for file in file_lists:
            if file != "old_train.csv" and os.path.isfile(
                os.path.join(TRAIN_FOLDER_PATH, file)
            ):
                print(file)
                os.remove(os.path.join(TRAIN_FOLDER_PATH, file))
        os.rename(os.path.join(TRAIN_FOLDER_PATH, "old_train.csv"), TRAIN_FILE_PATH)


def main(args):
    TRAIN_FILE_PATH = args.train_file_path
    TRAIN_FOLDER_PATH = args.train_folder_path
    if args.rollback:
        back2raw(TRAIN_FOLDER_PATH, TRAIN_FILE_PATH)
    else:
        raw_train_df = pd.read_csv(TRAIN_FILE_PATH)
        folds = StratifiedKFold(
            n_splits=FOLD_NUM, shuffle=True, random_state=SEED
        ).split(np.arange(raw_train_df.shape[0]), raw_train_df["label"])

        change_origin_file_name(TRAIN_FILE_PATH)
        for fold, (train_idx, test_idx) in enumerate(folds):
            if fold > 0:
                break
            train_ = raw_train_df.loc[train_idx, :].reset_index(drop=True)
            train_.to_csv(
                os.path.join(TRAIN_FOLDER_PATH, args.preprocessed_train_file_name),
                index=False,
            )

            test_ = raw_train_df.loc[test_idx, :].reset_index(drop=True)
            test_.to_csv(
                os.path.join(TRAIN_FOLDER_PATH, args.evaluation_file_name), index=False
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollback", type=bool, default=False, help="return to original data"
    )
    parser.add_argument(
        "--train_file_path",
        type=str,
        default="/opt/ml/dataset/train/train.csv",
        help="set train dataset file path (default : /opt/ml/dataset/train/train.csv)",
    )
    parser.add_argument(
        "--train_folder_path",
        type=str,
        default="/opt/ml/dataset/train",
        help="set train dataset folder path (default : /opt/ml/dataset/train)",
    )
    parser.add_argument(
        "--preprocessed_train_file_name",
        type=str,
        default="train.csv",
        help="set preprocessed train dataset file name (default : train.csv)",
    )
    parser.add_argument(
        "--evaluation_file_name",
        type=str,
        default="custom_test.csv",
        help="set evaluation dataset file name (default : custom_test.csv)",
    )

