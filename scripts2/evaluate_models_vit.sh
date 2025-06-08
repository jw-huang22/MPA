#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

DATASET1="cifar_10"
DATASET2="cifar_100"
DATASET3="food101"

python code/evaluate_model_vit.py --dataset "$DATASET1"
python code/evaluate_model_vit.py --dataset "$DATASET2"
python code/evaluate_model_vit.py --dataset "$DATASET3"

