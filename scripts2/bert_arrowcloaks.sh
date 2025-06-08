#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

GPUS="0,1"
DATASET1="mnli"
DATASET2="qqp"
DATASET3="qnli"
DATASET4="sst2"
WEIGHT_DIR="results/train_results"
RESTORE_DIR="results/arrowcloak_results"

./scripts/arrowcloak.sh --gpus $GPUS --dataset $DATASET1 --weight_dir $WEIGHT_DIR --restore_dir $RESTORE_DIR

./scripts/arrowcloak.sh --gpus $GPUS --dataset $DATASET2 --weight_dir $WEIGHT_DIR --restore_dir $RESTORE_DIR

./scripts/arrowcloak.sh --gpus $GPUS --dataset $DATASET3 --weight_dir $WEIGHT_DIR --restore_dir $RESTORE_DIR

./scripts/arrowcloak.sh --gpus $GPUS --dataset $DATASET4 --weight_dir $WEIGHT_DIR --restore_dir $RESTORE_DIR


echo "所有脚本执行完毕！"