#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/home/fit/renjuxjf/WORK/hjw/.cache/huggingface"
export HF_HUB_DISABLE_XET="1"
export HF_HUB_DOWNLOAD_TIMEOUT="60"
export HF_HUB_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"

DATASET1="cifar_10"
DATASET2="cifar_100"
DATASET3="food101"
OUTPUT_DIR="results/train_results/ViT"
TSQP_OUTPUT_DIR="results/tsqp_results/ViT"
TSQP="true"
OPT="adam"
LR="5e-5"
BS="64"
N_EPOCHS="10"
LR_SCHEDULER_TYPE="linear"
LR_SCHEDULER_WARMUP_RATIO="0.1"
WEIGHT_DECAY="0.01"
VIT_MODEL="vit_base_patch16_224.augreg_in21k"

./scripts/train_vit.sh --dataset $DATASET1 --output_dir $OUTPUT_DIR --opt $OPT --lr $LR --bs $BS --n_epochs $N_EPOCHS --lr_scheduler_type $LR_SCHEDULER_TYPE --lr_scheduler_warmup_ratio $LR_SCHEDULER_WARMUP_RATIO --weight_decay $WEIGHT_DECAY --vit_model $VIT_MODEL
./scripts/train_vit.sh --dataset $DATASET2 --output_dir $OUTPUT_DIR --opt $OPT --lr $LR --bs $BS --n_epochs $N_EPOCHS --lr_scheduler_type $LR_SCHEDULER_TYPE --lr_scheduler_warmup_ratio $LR_SCHEDULER_WARMUP_RATIO --weight_decay $WEIGHT_DECAY --vit_model $VIT_MODEL
./scripts/train_vit.sh --dataset $DATASET3 --output_dir $OUTPUT_DIR --opt $OPT --lr $LR --bs $BS --n_epochs $N_EPOCHS --lr_scheduler_type $LR_SCHEDULER_TYPE --lr_scheduler_warmup_ratio $LR_SCHEDULER_WARMUP_RATIO --weight_decay $WEIGHT_DECAY --vit_model $VIT_MODEL

# ./scripts/train_vit.sh --dataset $DATASET1 --output_dir $TSQP_OUTPUT_DIR --tsqp $TSQP
# ./scripts/train_vit.sh --dataset $DATASET2 --output_dir $TSQP_OUTPUT_DIR --tsqp $TSQP
# ./scripts/train_vit.sh --dataset $DATASET3 --output_dir $TSQP_OUTPUT_DIR --tsqp $TSQP


echo "所有脚本执行完毕！"
