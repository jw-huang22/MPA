#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

GPUS="0,1"

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

RESTORE_DIR="results/our_results"

for DATASET in "${DATASET_LIST[@]}"; do
    for OBFUS in "${OBFUS_LIST[@]}"; do
        ./scripts/mpa_gpt2.sh \
            --gpus "$GPUS" \
            --obfus "$OBFUS" \
            --dataset "$DATASET" \
            --restore_dir "$RESTORE_DIR"
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

# for DATASET in "${DATASET_LIST[@]}"; do
#     for RANK in "${RANK_LIST[@]}"; do
#         ./scripts/mpa_gpt2.sh \
#             --gpus "$GPUS" \
#             --obfus "AMO+arrowcloak" \
#             --dataset "$DATASET" \
#             --restore_dir "$RESTORE_DIR" \
#             --rank_r "$RANK"
#     done
# done


# ./scripts/mpa_gpt2.sh --gpus $GPUS --obfus AMO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 32
# ./scripts/mpa_gpt2.sh --gpus $GPUS --obfus AMO --dataset $DATASET2 --restore_dir $RESTORE_DIR --rank_r 32
# ./scripts/mpa_gpt2.sh --gpus $GPUS --obfus AMO --dataset $DATASET3 --restore_dir $RESTORE_DIR --rank_r 32
# ./scripts/mpa_gpt2.sh --gpus $GPUS --obfus AMO --dataset $DATASET4 --restore_dir $RESTORE_DIR --rank_r 32

# ./scripts/mpa_gpt2.sh --gpus $GPUS --obfus AMO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 0
# ./scripts/mpa_gpt2.sh --gpus $GPUS --obfus AMO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 1
# ./scripts/mpa_gpt2.sh --gpus $GPUS --obfus AMO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 2
# ./scripts/mpa_gpt2.sh --gpus $GPUS --obfus AMO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 4
# ./scripts/mpa_gpt2.sh --gpus $GPUS --obfus AMO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 8
# ./scripts/mpa_gpt2.sh --gpus $GPUS --obfus AMO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 16
# ./scripts/mpa_gpt2.sh --gpus $GPUS --obfus AMO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 32
# ./scripts/mpa_gpt2.sh --gpus $GPUS --obfus AMO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 64
# ./scripts/mpa_gpt2.sh --gpus $GPUS --obfus AMO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 128
# ./scripts/mpa_gpt2.sh --gpus $GPUS --obfus AMO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 256
# ./scripts/mpa_gpt2.sh --gpus $GPUS --obfus AMO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 512
# ./scripts/mpa_gpt2.sh --gpus $GPUS --obfus AMO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 768


echo "所有脚本执行完毕！"