#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

GPUS="0,1"
DATASET4="sst2"
output_dir1="results/train_results"
output_dir2="results/tsqp_results"
TSQP="true"

./scripts/train_gpt2_xl.sh --gpus $GPUS --dataset $DATASET4 --output_dir $output_dir1 
# ./scripts/train_gpt2_xl.sh --gpus $GPUS --dataset $DATASET4 --output_dir $output_dir2 --tsqp $TSQP


echo "所有脚本执行完毕！"