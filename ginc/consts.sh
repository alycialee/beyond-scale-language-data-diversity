#!/bin/bash

ROOT= #TODO
mkdir -p $ROOT

###### ADD CODE TO ACTIVATE ENVIRONMENT HERE #####


###########################

CACHE= #TODO
mkdir -p $CACHE
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE

LOGDIR=logs
mkdir -p $LOGDIR
