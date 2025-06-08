#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

GPUS="0,1"
MODEL="gpt2"
DATASET1="mnli"
DATASET2="qqp"
DATASET3="qnli"
DATASET4="sst2"
output_dir1="results/train_results"
output_dir2="results/tsqp_results"
TSQP="true"

./scripts/train.sh --gpus $GPUS --model $MODEL --dataset $DATASET1 --output_dir $output_dir1 
./scripts/train.sh --gpus $GPUS --model $MODEL --dataset $DATASET2 --output_dir $output_dir1 
./scripts/train.sh --gpus $GPUS --model $MODEL --dataset $DATASET3 --output_dir $output_dir1 
./scripts/train.sh --gpus $GPUS --model $MODEL --dataset $DATASET4 --output_dir $output_dir1 

./scripts/train.sh --gpus $GPUS --model $MODEL --dataset $DATASET1 --output_dir $output_dir2 --tsqp $TSQP
./scripts/train.sh --gpus $GPUS --model $MODEL --dataset $DATASET2 --output_dir $output_dir2 --tsqp $TSQP
./scripts/train.sh --gpus $GPUS --model $MODEL --dataset $DATASET3 --output_dir $output_dir2 --tsqp $TSQP
./scripts/train.sh --gpus $GPUS --model $MODEL --dataset $DATASET4 --output_dir $output_dir2 --tsqp $TSQP


echo "所有脚本执行完毕！"