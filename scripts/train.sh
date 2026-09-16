GPU_ID=0
export CUDA_VISIBLE_DEVICES=$GPU_ID
export TOKENIZERS_PARALLELISM=false

MODEL_NAME="OSD"

EXP_NAME="2026-1-15"
OUTPUT_BASE="outputs"
LOG_BASE="logs"

DATA_PATH="data/train/ff++,data/train/ff++"

OUTPUT_DIR="${OUTPUT_BASE}/${EXP_NAME}/${MODEL_NAME,,}"
LOG_DIR="${LOG_BASE}/${EXP_NAME}/${MODEL_NAME,,}"
MOE_CONFIG_PATH="configs/osd.json"
PY_ARGS=${@:1}

LAUNCHER="python"

$LAUNCHER main_finetune.py \
    --model $MODEL_NAME \
    --batch_size 32 \
    --blr 2e-3 \
    --epochs 10 \
    --num_workers 12 \
    --data_path "$DATA_PATH" \
    --output_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR/ \
    --moe_config_path "$MOE_CONFIG_PATH" \
    --training_mode "stage1_hard_sampling" \
    --artifact_aug True \
    --artifact_aug_dir "data/FaceForensics++/real-fake" \
    --save_ckpt_last_only True \
    ${PY_ARGS}

$LAUNCHER main_finetune.py \
    --model $MODEL_NAME \
    --batch_size 128 \
    --blr 1e-4 \
    --epochs 10 \
    --num_workers 12 \
    --data_path "$DATA_PATH" \
    --output_dir $OUTPUT_DIR\
    --log_dir $LOG_DIR \
    --moe_config_path "$MOE_CONFIG_PATH" \
    --training_mode "stage2_head_finetune" \
    --semantic_expert_scale 0.75 \
    --artifact_expert_scale 0.75 \
    --save_ckpt_last_only True \
    ${PY_ARGS}
