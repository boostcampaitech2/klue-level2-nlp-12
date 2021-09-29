#!/bin/bash
python train.py \
	--seed 42 \
	--model klue/bert-base \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --title tmp \
    # --run_kflod \
    # --fold_num \

    