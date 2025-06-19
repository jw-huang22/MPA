export HF_ENDPOINT="https://hf-mirror.com"

DATASET="cifar_100"
OUTPUT_DIR="results/train_results/ViT"
WEIGHT_DIR="results/train_results"
RESTORE_DIR="results/arrowcloak_results"

./scripts/train_vit.sh --dataset $DATASET --output_dir $OUTPUT_DIR

./scripts/arrowcloak_vit.sh  --dataset $DATASET --weight_dir $WEIGHT_DIR --restore_dir $RESTORE_DIR

echo "Successfully finished basic test2"