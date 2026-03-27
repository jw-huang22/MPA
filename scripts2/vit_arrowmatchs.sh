#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"

OBFUS1="translinkguard"
OBFUS2="tempo"
OBFUS3="soter"
OBFUS4="shadownet"
OBFUS5="tsqp"
OBFUS6="LoRO"
OBFUS7="obfuscatune"
DATASET1="cifar_10"
DATASET2="cifar_100"
DATASET3="food101"
RESTORE_DIR="results/arrowmatch_results"

# ./scripts/arrowmatch_vit.sh  --obfus $OBFUS1 --dataset $DATASET1 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch_vit.sh  --obfus $OBFUS2 --dataset $DATASET1 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch_vit.sh  --obfus $OBFUS3 --dataset $DATASET1 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch_vit.sh  --obfus $OBFUS4 --dataset $DATASET1 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch_vit.sh  --obfus $OBFUS5 --dataset $DATASET1 --restore_dir $RESTORE_DIR

# ./scripts/arrowmatch_vit.sh  --obfus $OBFUS1 --dataset $DATASET2 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch_vit.sh  --obfus $OBFUS2 --dataset $DATASET2 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch_vit.sh  --obfus $OBFUS3 --dataset $DATASET2 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch_vit.sh  --obfus $OBFUS4 --dataset $DATASET2 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch_vit.sh  --obfus $OBFUS5 --dataset $DATASET2 --restore_dir $RESTORE_DIR

./scripts/arrowmatch_vit.sh  --obfus $OBFUS1 --dataset $DATASET3 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch_vit.sh  --obfus $OBFUS2 --dataset $DATASET3 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch_vit.sh  --obfus $OBFUS3 --dataset $DATASET3 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch_vit.sh  --obfus $OBFUS4 --dataset $DATASET3 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch_vit.sh  --obfus $OBFUS5 --dataset $DATASET3 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch_vit.sh  --obfus $OBFUS6 --dataset $DATASET3 --restore_dir $RESTORE_DIR
# ./scripts/arrowmatch_vit.sh  --obfus $OBFUS7 --dataset $DATASET3 --restore_dir $RESTORE_DIR

echo "所有脚本执行完毕！"