#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

DATASET1="cifar_10"
DATASET2="cifar_100"
DATASET3="food101"
WEIGHT_DIR="results/train_results"
RESTORE_DIR="results/arrowcloak_results"

./scripts/arrowcloak_vit.sh  --dataset $DATASET1 --weight_dir $WEIGHT_DIR --restore_dir $RESTORE_DIR

./scripts/arrowcloak_vit.sh  --dataset $DATASET2 --weight_dir $WEIGHT_DIR --restore_dir $RESTORE_DIR

./scripts/arrowcloak_vit.sh  --dataset $DATASET3 --weight_dir $WEIGHT_DIR --restore_dir $RESTORE_DIR

echo "所有脚本执行完毕！"