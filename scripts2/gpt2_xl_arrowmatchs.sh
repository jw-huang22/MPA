export HF_ENDPOINT="https://hf-mirror.com"

GPUS="0,1"
OBFUS1="translinkguard"
OBFUS2="tempo"
OBFUS3="soter"
OBFUS4="shadownet"
OBFUS5="tsqp"
DATASET="sst2"
RESTORE_DIR="results/arrowmatch_results"

./scripts/arrowmatch_gpt2_xl.sh --gpus $GPUS --obfus $OBFUS1 --dataset $DATASET --restore_dir $RESTORE_DIR
./scripts/arrowmatch_gpt2_xl.sh --gpus $GPUS --obfus $OBFUS2 --dataset $DATASET --restore_dir $RESTORE_DIR
./scripts/arrowmatch_gpt2_xl.sh --gpus $GPUS --obfus $OBFUS3 --dataset $DATASET --restore_dir $RESTORE_DIR
./scripts/arrowmatch_gpt2_xl.sh --gpus $GPUS --obfus $OBFUS4 --dataset $DATASET --restore_dir $RESTORE_DIR
./scripts/arrowmatch_gpt2_xl.sh --gpus $GPUS --obfus $OBFUS5 --dataset $DATASET --restore_dir $RESTORE_DIR


echo "所有脚本执行完毕！"