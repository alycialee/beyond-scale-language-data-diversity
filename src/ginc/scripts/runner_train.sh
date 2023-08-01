#!/bin/bash

source ./consts.sh
chmod +x ./scripts/*.sh

export CUDA_VISIBLE_DEVICES=2

wandb agent ginc-models/qh98d82r