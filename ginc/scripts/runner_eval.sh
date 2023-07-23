#!/bin/bash
source ./consts.sh

export CUDA_VISIBLE_DEVICES=0

# Specify trained model to run evaluation on.
OUTPUT_DIR="../ginc-output/train/mid_model"
NLAYERS=12
NHMMS=20
NSYMBOLS=50
NUM_EX=1000
CHECKPOINT=6000

# Evaluate on GINC data with specified NHMMSFORDATA and NSYMBOLSFORDATA.
NHMMSFORDATA=2
NSYMBOLSFORDATA=50

for NHMMS in 200 500 1000 2000 10000 20000
do
for NHMMSFORDATA in 2 200 20000
do
for SEED in 1111 1112 1113 1114 1115
do
MODEL_FOLDER=GINC_trans0.1_start10.0_nsymbols${NSYMBOLS}_nvalues10_nslots10_vic0.9_nhmms${NHMMS}_seed1111_nlayers=${NLAYERS}_dembed=768_trainseed=${SEED}
MODEL_DIR=$OUTPUT_DIR/$MODEL_FOLDER/checkpoint-${CHECKPOINT}

python run_eval.py \
    --model_name_or_path $MODEL_DIR \
    --tokenizer_name ${DATA_DIR}/tokenizer.json \
    --custom_tokenizer \
    --small_model \
    --output_dir=. \
    --eval_incontext \
    --logging_steps 100 \
    --save_total_limit 2 \
    --evaluation_strategy steps \
    --train_file=tmp.json \
    --validation_file=tmp.json \
    --block_size 1024 \
    --learning_rate 8e-4 \
    --num_train_epochs 3 \
    --warmup_steps 1000 \
    --fp16 \
    --custom_num_layers $NLAYERS \
    --n_hmms $NHMMSFORDATA \
    --n_symbols $NSYMBOLSFORDATA \
    --dataset_seed 1111
done
done
done