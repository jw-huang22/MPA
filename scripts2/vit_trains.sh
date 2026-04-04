#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

DATASET1="cifar_10"
DATASET2="cifar_100"
DATASET3="food101"
OUTPUT_DIR1="results/train_results/ViT"
OUTPUT_DIR2="results/tsqp_results/ViT"
OUTPUT_DIR3="results/train_results/ViT_adam"
TSQP="true"
OPT="adam"

# ./scripts/train_vit.sh --dataset $DATASET1 --output_dir $OUTPUT_DIR1
# ./scripts/train_vit.sh --dataset $DATASET2 --output_dir $OUTPUT_DIR1
# ./scripts/train_vit.sh --dataset $DATASET3 --output_dir $OUTPUT_DIR1

# ./scripts/train_vit.sh --dataset $DATASET1 --output_dir $OUTPUT_DIR2 --tsqp $TSQP
# ./scripts/train_vit.sh --dataset $DATASET2 --output_dir $OUTPUT_DIR2 --tsqp $TSQP
# ./scripts/train_vit.sh --dataset $DATASET3 --output_dir $OUTPUT_DIR2 --tsqp $TSQP


./scripts/train_vit.sh --dataset $DATASET3 --output_dir $OUTPUT_DIR3 --opt $OPT

echo "所有脚本执行完毕！"