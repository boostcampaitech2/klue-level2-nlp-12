#!/bin/bash
python train.py \
	--seed 42 \
	--model klue/roberta-large \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --title tmp \
    # --run_kflod \
    # --fold_num \

    