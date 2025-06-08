#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

DATASET="sst2"
GPUS="0,1"

python code/evaluate_model_gpt2_xl.py --dataset "$DATASET" --gpus "$GPUS" 

