#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

GPUS="0,1"
DATASET="sst2"
WEIGHT_DIR="results/train_results"
RESTORE_DIR="results/arrowcloak_results"

./scripts/arrowcloak_gpt2_xl.sh --gpus $GPUS --dataset $DATASET --weight_dir $WEIGHT_DIR --restore_dir $RESTORE_DIR


echo "所有脚本执行完毕！"