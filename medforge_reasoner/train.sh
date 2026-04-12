#!/bin/bash

# Get absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_FILE="${BASH_SOURCE[0]}"

# Get project root directory (assume script is under deepfake_localization/scripts/bbox/)
# Search upwards from script path for project root
PROJECT_ROOT="$SCRIPT_DIR"
while [ ! -d "$PROJECT_ROOT/output" ] && [ "$PROJECT_ROOT" != "/" ]; do
    PROJECT_ROOT=$(dirname "$PROJECT_ROOT")
done

# If project root not found, use grandparent of script directory
if [ "$PROJECT_ROOT" == "/" ] || [ ! -d "$PROJECT_ROOT/output" ]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi

# Base output directory
BASE_OUTPUT_DIR="./output/sft/Qwen3vl-8B-Instruct"

# Check if script is already running within a version directory
# If script's parent directory is BASE_OUTPUT_DIR, it's already in a version folder
SCRIPT_PARENT=$(dirname "$SCRIPT_DIR")
if [ "$SCRIPT_PARENT" == "$BASE_OUTPUT_DIR" ] && [[ $(basename "$SCRIPT_DIR") =~ ^v[0-9]+- ]]; then
    # Already in a version directory, use current directory
    VERSION_DIR="$SCRIPT_DIR"
    echo "Script is already in a version directory: $VERSION_DIR"
    echo "Using existing directory as output."
else
    # Ensure base output directory exists
    mkdir -p "$BASE_OUTPUT_DIR"

    # Find maximum existing version number
    MAX_VERSION=0
    if [ -d "$BASE_OUTPUT_DIR" ]; then
        for dir in "$BASE_OUTPUT_DIR"/v*-*; do
            if [ -d "$dir" ]; then
                # Extract version number (digits after 'v')
                dirname=$(basename "$dir")
                if [[ $dirname =~ ^v([0-9]+)- ]]; then
                    version=${BASH_REMATCH[1]}
                    if [ "$version" -gt "$MAX_VERSION" ]; then
                        MAX_VERSION=$version
                    fi
                fi
            fi
        done
    fi

    # Generate new version number
    NEW_VERSION=$((MAX_VERSION + 1))

    # Generate timestamp (format: YYYYMMDD-HHMMSS)
    TIMESTAMP=$(date +"%Y%m%d-%H%M%S")

    # Create new version directory
    VERSION_DIR="$BASE_OUTPUT_DIR/v${NEW_VERSION}-${TIMESTAMP}"
    mkdir -p "$VERSION_DIR"

    # Copy script to the new directory
    cp "$SCRIPT_FILE" "$VERSION_DIR/train.sh"

    echo "Created new version directory: $VERSION_DIR"
    echo "Script copied to: $VERSION_DIR/train.sh"
fi

# 2 * 21GiB
# Note: Using qwen3_vl_resnet model will automatically use default path:
# ./output/best_model.pth
# If another path is needed, set environment variable:
# export RESNET_CKPT="/path/to/your/best_model.pth"
MASTER_PORT=29502 \
WANDB_API_KEY=your_wandb_api_key \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model "path/to/your/Qwen3-VL-8B-Instruct" \
    --dataset 'data/sft_train.json' \
    --split_dataset_ratio 0.05 \
    --load_from_cache_file true \
    --train_type lora \
    --lora_rank 128 \
    --lora_alpha 256 \
    --torch_dtype bfloat16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --attn_impl flash_attn \
    --learning_rate 1e-4 \
    --target_modules all-linear \
    --freeze_vit false \
    --freeze_aligner false \
    --packing false \
    --padding_free true \
    --gradient_checkpointing  \
    --vit_gradient_checkpointing true \
    --gradient_accumulation_steps 4 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 6 \
    --max_length 2048 \
    --output_dir "$VERSION_DIR" \
    --warmup_ratio 0.05 \
    --deepspeed zero2 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --report_to wandb \
