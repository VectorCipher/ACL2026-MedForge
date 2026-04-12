#!/bin/bash

# Swift Rollout Server Startup Script
# This script should be run on the remote server that will handle inference/rollout
# for GRPO training.

# ============================================================
# CONFIGURATION
# ============================================================

# Model Configuration
# BASE_MODEL="path/to/your/model"
BASE_MODEL_NAME="Qwen3vl-8B-Instruct"
# BASE_MODEL="path/to/another/model"
# BASE_MODEL_NAME="MiMo-VL-7B"
BASE_MODEL="./output/sft/Qwen3vl-8B-Instruct-merged"
BASE_MODEL_NAME="Qwen3vl-8B-Instruct"
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

# Server Configuration
PORT=8000
GPUS="6,7"  # GPUs to use for rollout
IFS=',' read -ra GPU_ARR <<< "$GPUS"
VLLM_DATA_PARALLEL_SIZE=${#GPU_ARR[@]}  # Should match number of GPUs

# ============================================================
# START ROLLOUT SERVER
# ============================================================

echo "=========================================="
echo "Starting SWIFT Rollout Server"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "GPUs: $GPUS"
echo "Data Parallel Size: $VLLM_DATA_PARALLEL_SIZE"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPUS \
swift rollout \
    --model "$MODEL_PATH" \
    --vllm_data_parallel_size $VLLM_DATA_PARALLEL_SIZE \
    --port $PORT \
    --vllm_tensor_parallel_size 1 \
    --vllm_enable_prefix_caching false \
    --max_new_tokens 4096 \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 12288 \
    --temperature 1.0 \
    --served_model_name $BASE_MODEL_NAME \
    --vllm_enable_lora false \
    --vllm_limit_mm_per_prompt '{"image": 1, "video": 0}' \

echo ""
echo "Rollout server stopped."

