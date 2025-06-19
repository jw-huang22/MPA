#!/bin/bash
 
export HF_ENDPOINT="https://hf-mirror.com"

GPUS="0,1"
MODEL="bert-base-cased"
DATASET="sst2"
output_dir="results/train_results"
WEIGHT_DIR="results/train_results"
RESTORE_DIR="results/arrowcloak_results"

./scripts/train.sh --gpus $GPUS --model $MODEL --dataset $DATASET --output_dir $output_dir

original_file_name="$output_dir/bert/sst2/checkpoint-3159"
new_file_name="$output_dir/bert/sst2/final_checkpoint"
mv "$original_file_name" "$new_file_name"

./scripts/arrowcloak.sh --gpus $GPUS --dataset $DATASET --weight_dir $WEIGHT_DIR --restore_dir $RESTORE_DIR

original_file_name="$RESTORE_DIR/bert/arrowcloak/sst2/checkpoint-33"
new_file_name="$RESTORE_DIR/bert/arrowcloak/sst2/final_checkpoint"
mv "$original_file_name" "$new_file_name"

echo "Successfully finished basic test1"