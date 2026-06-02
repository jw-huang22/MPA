#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

GPUS="0,1"
RESTORE_DIR="results/arrowmatch_results"
RANK_R="8"

OBFUS_LIST=(
    # "black"
    # "translinkguard"
    # "tempo"
    # "soter"
    # "shadownet"
    # "LoRO"
    # "obfuscatune"
    # "groupcover"
    "twinshield"
    # "arrowcloak"
)

DATASET_LIST=(
    "mnli"
    "qqp"
    "qnli"
    "sst2"
)

for DATASET in "${DATASET_LIST[@]}"; do
    for OBFUS in "${OBFUS_LIST[@]}"; do
        ./scripts/arrowmatch_gpt2.sh \
            --gpus "$GPUS" \
            --obfus "$OBFUS" \
            --dataset "$DATASET" \
            --restore_dir "$RESTORE_DIR" \
            --rank_r "$RANK_R"
    done
done

RANK_LIST=(
    # "1"
    # "2"
    # "4"
    # "8"
    # "16"
    "32"
    "64"
    "128"
    # "256"
    # "512"
    # "768"
)

for RANK in "${RANK_LIST[@]}"; do
    for DATASET in "${DATASET_LIST[@]}"; do
        ./scripts/arrowmatch_gpt2.sh \
            --gpus "$GPUS" \
            --obfus "AMO+arrowcloak" \
            --dataset "$DATASET" \
            --restore_dir "$RESTORE_DIR" \
            --rank_r "$RANK"
    done
done

echo "所有脚本执行完毕！"
