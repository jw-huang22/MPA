#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

DATASET1="cifar_10"
DATASET2="cifar_100"
DATASET3="food101"
OUTPUT_DIR1="results/train_results/ViT"
OUTPUT_DIR2="results/tsqp_results/ViT"
TSQP="true"

./scripts/train_vit.sh --dataset $DATASET1 --output_dir $OUTPUT_DIR1
./scripts/train_vit.sh --dataset $DATASET2 --output_dir $OUTPUT_DIR1
./scripts/train_vit.sh --dataset $DATASET3 --output_dir $OUTPUT_DIR1

./scripts/train_vit.sh --dataset $DATASET1 --output_dir $OUTPUT_DIR2 --tsqp $TSQP
./scripts/train_vit.sh --dataset $DATASET2 --output_dir $OUTPUT_DIR2 --tsqp $TSQP
./scripts/train_vit.sh --dataset $DATASET3 --output_dir $OUTPUT_DIR2 --tsqp $TSQP


echo "所有脚本执行完毕！"