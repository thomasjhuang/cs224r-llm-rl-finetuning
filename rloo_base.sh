#!/bin/bash

# RLOO Performance Experiment
# CS224R Final Project - RL Fine-tuning of Language Models

set -e  # Exit on any error

# Activate conda environment
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate cs224r-project

echo "Starting RLOO Performance Experiment"
echo "============================================================"

# Use the custom SFT warmup model as the starting point
SFT_MODEL_PATH="./sft_warmup_model"
DATASET_NAME="Jiayi-Pan/Countdown-Tasks-3to4"
OUTPUT_DIR="./rloo_run_with_custom_sft"
WANDB_PROJECT="qwen2-countdown-rloo"
WANDB_RUN_NAME="rloo-k8-s500-T0.8-custom-sft"

# Training hyperparameters for performance
K_SAMPLES=8
LR=5e-6
BATCH_SIZE=2
NUM_EPOCHS=3
MAX_STEPS=500
GRAD_ACCUM=4
MAX_LENGTH_GEN=100
TEMPERATURE=0.8
SAVE_STEPS=100
LOG_STEPS=5

# TTC related (disabled by default)
TTC_INTERNAL_SAMPLES_N=1 # Keep at 1 to disable TTC
TTC_LAMBDA_COST=0.0    # Keep at 0.0 to disable TTC

echo "------------------------------------------------------------"
echo "Run Parameters:"
echo "  SFT Model: $SFT_MODEL_PATH"
echo "  Dataset: $DATASET_NAME"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Run Name: $WANDB_RUN_NAME"
echo "  k_samples: $K_SAMPLES"
echo "  Max Steps: $MAX_STEPS"
echo "  Temperature: $TEMPERATURE"
echo "  TTC: Disabled"
echo "------------------------------------------------------------"

python src/main_rloo.py \
    --sft_model_path "$SFT_MODEL_PATH" \
    --prompt_dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --k_samples $K_SAMPLES \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --max_steps $MAX_STEPS \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --max_length_generation $MAX_LENGTH_GEN \
    --temperature $TEMPERATURE \
    --log_every_n_steps $LOG_STEPS \
    --save_every_n_steps $SAVE_STEPS \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run_name "$WANDB_RUN_NAME" \
    --ttc_internal_samples_n $TTC_INTERNAL_SAMPLES_N \
    --ttc_lambda_cost $TTC_LAMBDA_COST

echo "RLOO training finished. Model saved to $OUTPUT_DIR" 