PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift infer \
    --model "path/to/your/merged_checkpoint" \
    --adapters "path/to/your/adapter_checkpoint" \
    --val_dataset 'data/test_dataset.json' \
    --stream false \
    --infer_backend pt \
    --max_batch_size 64 \
    --max_new_tokens 2048 \
    --load_data_args false \
    --metric rouge \
    --temperature 0 \
    --top_k 1 \
    --result_path ./output/test_results.jsonl \