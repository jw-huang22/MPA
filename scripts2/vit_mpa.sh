#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/home/fit/renjuxjf/WORK/hjw/.cache/huggingface"
export HF_HUB_DISABLE_XET="1"
export HF_HUB_DOWNLOAD_TIMEOUT="60"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export PYTHONUNBUFFERED=1
export OPENBLAS_CORETYPE=HASWELL

GPUS="0"
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
    # "AMO"
    # "AMO+shadownet"
)

DATASET_LIST=(
    "cifar_10"
    "cifar_100"
    "food101"
)

RESTORE_DIR="results/our_results"

for DATASET in "${DATASET_LIST[@]}"; do
    for OBFUS in "${OBFUS_LIST[@]}"; do
        ./scripts/mpa_vit.sh \
            --gpus "$GPUS" \
            --obfus "$OBFUS" \
            --dataset "$DATASET"\
            --restore_dir "$RESTORE_DIR" \
            --rank_r 32
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
#         ./scripts/mpa_vit.sh \
#             --gpus "$GPUS" \
#             --obfus "AMO+arrowcloak" \
#             --dataset "$DATASET"\
#             --restore_dir "$RESTORE_DIR" \
#             --rank_r "$RANK"
#     done
# done


# ./scripts/mpa_vit.sh --gpus $GPUS --obfus AMO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 32 
# ./scripts/mpa_vit.sh --gpus $GPUS --obfus AMO --dataset $DATASET2 --restore_dir $RESTORE_DIR --rank_r 32
# ./scripts/mpa_vit.sh --gpus $GPUS --obfus AMO --dataset $DATASET3 --restore_dir $RESTORE_DIR --rank_r 32


# Rank sweeps, aligned with bert_mpa.sh/gpt2_mpa.sh. Uncomment as needed.
# ./scripts/mpa_vit.sh --gpus $GPUS --obfus AMO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 0
# ./scripts/mpa_vit.sh --gpus $GPUS --obfus AMO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 2
# ./scripts/mpa_vit.sh --gpus $GPUS --obfus AMO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 4
# ./scripts/mpa_vit.sh --gpus $GPUS --obfus AMO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 8
# ./scripts/mpa_vit.sh --gpus $GPUS --obfus AMO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 16
# ./scripts/mpa_vit.sh --gpus $GPUS --obfus AMO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 32
# ./scripts/mpa_vit.sh --gpus $GPUS --obfus AMO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 64
# ./scripts/mpa_vit.sh --gpus $GPUS --obfus LoRO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 0
# ./scripts/mpa_vit.sh --gpus $GPUS --obfus LoRO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 2
# ./scripts/mpa_vit.sh --gpus $GPUS --obfus LoRO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 4
# ./scripts/mpa_vit.sh --gpus $GPUS --obfus LoRO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 8
# ./scripts/mpa_vit.sh --gpus $GPUS --obfus LoRO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 16
# ./scripts/mpa_vit.sh --gpus $GPUS --obfus LoRO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 32
# ./scripts/mpa_vit.sh --gpus $GPUS --obfus LoRO --dataset $DATASET1 --restore_dir $RESTORE_DIR --rank_r 64

echo "所有脚本执行完毕！"
