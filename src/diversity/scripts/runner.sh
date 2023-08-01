#!/bin/bash

chmod +x *.sh
chmod +x ./scripts/*.sh

export CUDA_VISIBLE_DEVICES=0

DATASET=c4
NUM_TASKS=200
BATCH_SIZE=512
BUFFER_SIZE=500000
CHECKPOINT=6000
OUTPUT_DIR="../div-output"
CACHE_DIR="../cache_dir"

for DATASET in c4 wikitext the_pile
do
python main.py \
    --task_name $DATASET \
    --num_tasks $NUM_TASKS \
    --batch_size $BATCH_SIZE \
    --buffer_size $BUFFER_SIZE \
    --finetune \
    --pretrained \
    --output_dir ${OUTPUT_DIR}/output_${DATASET}_${NUM_TASKS}tasks_bs${BATCH_SIZE}_gpt2_pt_ft \
    --cache_dir $CACHE_DIR
done