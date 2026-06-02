#!/bin/bash

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HOME="/home/fit/renjuxjf/WORK/hjw/.cache/huggingface"
export HF_HUB_DISABLE_XET="1"
export HF_HUB_DOWNLOAD_TIMEOUT="300"

unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_DATASETS_OFFLINE

VIT_MODEL="${VIT_MODEL:-vit_base_patch16_224.augreg_in21k}"

python - <<'PY'
import os

import timm

model_name = os.environ.get("VIT_MODEL", "vit_base_patch16_224.augreg_in21k")
print(f"Downloading/caching timm model: {model_name}")
print(f"HF_HOME={os.environ.get('HF_HOME')}")

timm.create_model(model_name, pretrained=True)

print("ViT pretrained weights are cached.")
PY
