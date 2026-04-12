#!/bin/bash

# GRPO Training Script

# ============================================================
# CONFIGURATION SECTION
# ============================================================

# --- Model Configuration ---
BASE_MODEL="./output/sft/Qwen3vl-8B-Instruct-merged"
# ADAPTER_PATH="path/to/your/sft/checkpoint"
# MERGED_MODEL_PATH="${ADAPTER_PATH}-merged"
# BASE_MODEL="path/to/another/model"

# Choose model
USE_BASE_MODEL=true

if [ "$USE_BASE_MODEL" = true ]; then
    MODEL_PATH="$BASE_MODEL"
else
    # Merge LoRA if needed
    if [ ! -d "$MERGED_MODEL_PATH" ]; then
        echo "Merging LoRA adapter from $ADAPTER_PATH to $MERGED_MODEL_PATH ..."
        swift export \
            --ckpt_dir "$ADAPTER_PATH" \
            --merge_lora true \
            --output_dir "$MERGED_MODEL_PATH" \
            --torch_dtype bfloat16
    else
        echo "Merged model found at $MERGED_MODEL_PATH"
    fi
    MODEL_PATH="$MERGED_MODEL_PATH"
fi


# --- Dataset and Output ---
DATASET_PATH="data/grpo_train.json"

# Get absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_FILE="${BASH_SOURCE[0]}"

# Base output directory
BASE_OUTPUT_DIR="./output/grpo/Qwen3vl-8B-Instruct"

# Check if script is already running within a version directory
SCRIPT_PARENT=$(dirname "$SCRIPT_DIR")
if [ "$SCRIPT_PARENT" == "$BASE_OUTPUT_DIR" ] && [[ $(basename "$SCRIPT_DIR") =~ ^v[0-9]+- ]]; then
    # Already in a version directory, use current directory
    VERSION_DIR="$SCRIPT_DIR"
    echo "Script is already in a version directory: $VERSION_DIR"
    echo "Using existing directory as output."
else
    # Ensure base output directory exists
    mkdir -p "$BASE_OUTPUT_DIR"

    # Find the maximum existing version number
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

    # Copy script to the new directory, keeping the name consistent
    SCRIPT_NAME=$(basename "$SCRIPT_FILE")
    cp "$SCRIPT_FILE" "$VERSION_DIR/$SCRIPT_NAME"

    echo "Created new version directory: $VERSION_DIR"
    echo "Script copied to: $VERSION_DIR/$SCRIPT_NAME"
fi

# Output directory for checkpoints and logs
OUTPUT_DIR="$VERSION_DIR"

# Plugin file containing custom reward function
PLUGIN_PATH="./grpo_plugin_reward_coverage.py"

# --- Training Configuration ---
TRAIN_DEVICES="0,1,2,3,4,5"  # GPUs for training, use CUDA devices 0-5
IFS=',' read -ra GPU_ARR <<< "$TRAIN_DEVICES"
NPROC_PER_NODE=${#GPU_ARR[@]}  # Number of training GPUs
echo "[DEBUG] NPROC_PER_NODE: $NPROC_PER_NODE"
echo "[DEBUG] GPU_ARR: ${GPU_ARR[@]}"
echo "[DEBUG] TRAIN_DEVICES: $TRAIN_DEVICES"

# ============================================================
# MAIN EXECUTION
# ============================================================

echo "=========================================="
echo "GRPO Training"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Training GPUs: $TRAIN_DEVICES (${NPROC_PER_NODE} GPUs)"
echo "=========================================="

echo ""
echo "Starting GRPO Training..."
echo ""

export PORT=8000
export CUDA_VISIBLE_DEVICES=$TRAIN_DEVICES

# Set PyTorch distributed training environment variables
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export TORCH_DISTRIBUTED_DEBUG=INFO 

WANDB_API_KEY=your_wandb_api_key \
NPROC_PER_NODE=$NPROC_PER_NODE \
swift rlhf \
    --rlhf_type grpo \
    --model "$MODEL_PATH" \
    --dataset "$DATASET_PATH" \
    --external_plugins "$PLUGIN_PATH" \
    --reward_funcs deepfake_complex_reward \
    --split_dataset_ratio 0.01 \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_enable_lora true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port $PORT \
    --train_type lora \
    --freeze_vit false \
    --freeze_aligner false \
    --lora_rank 128 \
    --lora_alpha 256 \
    --torch_dtype bfloat16 \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 16 \
    --save_strategy steps \
    --eval_strategy steps \
    --eval_steps 10 \
    --save_steps 10 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --num_generations 8 \
    --temperature 1.0 \
    --beta 0.001 \
    --max_grad_norm 0.5 \
    --deepspeed zero3 \
    --log_completions true \
    --report_to wandb \
    --num_iterations 1 \
    --async_generate false \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --steps_per_generation 4 \
    --beta 0 \
    --importance_sampling_level sequence \

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="

