#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

GPUS="0,1"
OBFUS0="black"
OBFUS1="translinkguard"
OBFUS2="tempo"
OBFUS3="soter"
OBFUS4="shadownet"
OBFUS5="tsqp"
OBFUS6="arrowcloak"
OBFUS7="LoRO"
OBFUS8="obfuscatune"
OBFUS9="groupcover"
OBFUS10="twinshield"
DATASET="sst2"
RESTORE_DIR="results/our_results"

# ./scripts/mpa_gpt2_xl.sh --gpus $GPUS --obfus $OBFUS0 --dataset $DATASET --restore_dir $RESTORE_DIR
# ./scripts/mpa_gpt2_xl.sh --gpus $GPUS --obfus $OBFUS1 --dataset $DATASET --restore_dir $RESTORE_DIR
# ./scripts/mpa_gpt2_xl.sh --gpus $GPUS --obfus $OBFUS2 --dataset $DATASET --restore_dir $RESTORE_DIR
# ./scripts/mpa_gpt2_xl.sh --gpus $GPUS --obfus $OBFUS3 --dataset $DATASET --restore_dir $RESTORE_DIR
# ./scripts/mpa_gpt2_xl.sh --gpus $GPUS --obfus $OBFUS4 --dataset $DATASET --restore_dir $RESTORE_DIR
# ./scripts/mpa_gpt2_xl.sh --gpus $GPUS --obfus $OBFUS5 --dataset $DATASET --restore_dir $RESTORE_DIR
# ./scripts/mpa_gpt2_xl.sh --gpus $GPUS --obfus $OBFUS6 --dataset $DATASET --restore_dir $RESTORE_DIR
# ./scripts/mpa_gpt2_xl.sh --gpus $GPUS --obfus $OBFUS7 --dataset $DATASET --restore_dir $RESTORE_DIR
# ./scripts/mpa_gpt2_xl.sh --gpus $GPUS --obfus $OBFUS8 --dataset $DATASET --restore_dir $RESTORE_DIR
./scripts/mpa_gpt2_xl.sh --gpus $GPUS --obfus $OBFUS10 --dataset $DATASET --restore_dir $RESTORE_DIR

# ./scripts/mpa_gpt2_xl.sh --gpus $GPUS --obfus "AMO+arrowcloak" --dataset $DATASET --restore_dir $RESTORE_DIR --rank_r 32
# ./scripts/mpa_gpt2_xl.sh --gpus $GPUS --obfus "AMO+arrowcloak" --dataset $DATASET --restore_dir $RESTORE_DIR --rank_r 64
# ./scripts/mpa_gpt2_xl.sh --gpus $GPUS --obfus "AMO+arrowcloak" --dataset $DATASET --restore_dir $RESTORE_DIR --rank_r 128


# ./scripts/mpa_gpt2_xl.sh --gpus $GPUS --obfus $OBFUS9 --dataset $DATASET --restore_dir $RESTORE_DIR

echo "所有脚本执行完毕！"
