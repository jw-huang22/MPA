#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

DATASET="cifar_100"
OUTPUT_DIR="results/train_results/ViT"
TSQP="false"
OPT="sgd"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --tsqp) TSQP="$2"; shift ;;
        --opt) OPT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

python train/train_vit.py \
    --dataset "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --tsqp "$TSQP" \
    --opt "$OPT"