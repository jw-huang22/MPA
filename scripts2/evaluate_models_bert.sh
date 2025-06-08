#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

DATASET1="mnli"
DATASET2="qqp"
DATASET3="qnli"
DATASET4="sst2"
GPUS="0,1"

python code/evaluate_model.py --dataset "$DATASET1"  --gpus "$GPUS" 
python code/evaluate_model.py --dataset "$DATASET2"  --gpus "$GPUS" 
python code/evaluate_model.py --dataset "$DATASET3"  --gpus "$GPUS" 
python code/evaluate_model.py --dataset "$DATASET4"  --gpus "$GPUS" 

