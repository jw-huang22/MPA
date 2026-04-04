#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

GPUS="0,1"
OBFUS0="black"
OBFUS1="translinkguard"
OBFUS2="tempo"
OBFUS3="soter"
OBFUS4="shadownet"
OBFUS5="tsqp"
OBFUS6="LoRO"
OBFUS7="obfuscatune"
OBFUS8="groupcover"
OBFUS9="twinshield"
OBFUS10="arrowcloak"
DATASET1="mnli"
DATASET2="qqp"
DATASET3="qnli"
DATASET4="sst2"
RESTORE_DIR="results/our_results"


# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS0 --dataset $DATASET1 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS1 --dataset $DATASET1 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS2 --dataset $DATASET1 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS3 --dataset $DATASET1 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS4 --dataset $DATASET1 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS5 --dataset $DATASET1 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS6 --dataset $DATASET1 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS7 --dataset $DATASET1 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS8 --dataset $DATASET1 --restore_dir $RESTORE_DIR
./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS9 --dataset $DATASET1 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS10 --dataset $DATASET1 --restore_dir $RESTORE_DIR

# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS1 --dataset $DATASET2 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS2 --dataset $DATASET2 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS3 --dataset $DATASET2 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS4 --dataset $DATASET2 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS5 --dataset $DATASET2 --restore_dir $RESTORE_DIR

# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS1 --dataset $DATASET3 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS2 --dataset $DATASET3 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS3 --dataset $DATASET3 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS4 --dataset $DATASET3 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS5 --dataset $DATASET3 --restore_dir $RESTORE_DIR

# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS0 --dataset $DATASET4 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS1 --dataset $DATASET4 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS2 --dataset $DATASET4 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS3 --dataset $DATASET4 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS4 --dataset $DATASET4 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS5 --dataset $DATASET4 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS6 --dataset $DATASET4 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS7 --dataset $DATASET4 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS8 --dataset $DATASET4 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS9 --dataset $DATASET4 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch.sh --gpus $GPUS --obfus $OBFUS10 --dataset $DATASET4 --restore_dir $RESTORE_DIR

echo "所有脚本执行完毕！"