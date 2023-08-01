#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

BATCH_SIZE=512
N_HMMS=10
N_SYMBOLS=50

for N_HMMS in 2 10 20 50 100
do
for N_SYMBOLS in 50 100 150
do
python main_ginc.py \
    --batch_size $BATCH_SIZE \
    --finetune \
    --pretrained \
    --cache_dir cache_dir \
    --n_hmms $N_HMMS \
    --n_symbols $N_SYMBOLS \
done
done