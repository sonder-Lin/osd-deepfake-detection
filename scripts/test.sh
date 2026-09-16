#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export TOKENIZERS_PARALLELISM=false

DATASET="${1:-celebdf}"

python main_finetune.py \
    --model OSD \
    --batch_size 2048 \
    --num_workers 8 \
    --eval_data_path "data/test/${DATASET}" \
    --output_dir "outputs/2026-1-18/osd/eval/${DATASET}" \
    --resume "outputs/2026-1-15/osd/head_finetune/checkpoint-last.pth" \
    --moe_config_path configs/osd.json \
    --eval True \
    --use_gt_router False \
    --semantic_expert_scale 0.5 \
    --artifact_expert_scale 0.5
