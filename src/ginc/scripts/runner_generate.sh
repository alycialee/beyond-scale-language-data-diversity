#!/bin/bash

source ./consts.sh
chmod +x *.sh
chmod +x ./scripts/*.sh

export CUDA_VISIBLE_DEVICES=0

N_VALUES=10
N_SLOTS=10
N_SYMBOLS=50
TRANS_TEMP=0.1
START_TEMP=10.0
VIC=0.9
N_HMMS=10
SEED=1111
NUM_EX=1000

for N_HMMS in 2 10 20 50 100, 200 500 1000 2000 10000 20000
do
for N_SYMBOLS in 50 100 150 250 500 1000 5000 10000 20000 50000
do
for NUM_EX in 1000
do
python generate_data.py \
    --transition_temp $TRANS_TEMP \
    --start_temp $START_TEMP \
    --n_symbols $N_SYMBOLS \
    --n_values $N_VALUES \
    --n_slots $N_SLOTS \
    --value_identity_coeff $VIC \
    --n_hmms $N_HMMS \
    --root $ROOT \
    --seed $SEED \
    --n_train_samples $NUM_EX
done
done
done