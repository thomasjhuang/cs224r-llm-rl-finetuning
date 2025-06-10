#!/bin/bash

# RLOO Optimized Experiment
# CS224R Final Project - RL Fine-tuning of Language Models

set -e  # Exit on any error

# Activate conda environment
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate cs224r-project

echo "Starting RLOO Optimized Experiment"
echo "============================================================"

# Use HuggingFace uploaded model
SFT_MODEL_PATH="thomasjhuang/qwen2-sft-warmup"
DATASET_NAME="Jiayi-Pan/Countdown-Tasks-3to4"
OUTPUT_DIR="./rloo_optimized_run"
WANDB_PROJECT="qwen2-countdown-rloo"
WANDB_RUN_NAME="rloo-k8-best_of_16-T0.8"

# Training hyperparameters for performance
K_SAMPLES=8
LR=5e-6
BATCH_SIZE=8
MAX_STEPS=500
GRAD_ACCUM_STEPS=4
MAX_LENGTH_GEN=100
TEMPERATURE=0.8

# TTC ENABLED for Best-of-N sampling
TTC_INTERNAL_SAMPLES=16
TTC_LAMBDA_COST=0.01

# Logging
LOG_EVERY=1
SAVE_EVERY=50

echo "Configuration:"
echo "  Model: ${SFT_MODEL_PATH}"
echo "  Dataset: ${DATASET_NAME}"
echo "  Output: ${OUTPUT_DIR}"
echo "  k_samples: ${K_SAMPLES}"
echo "  Learning Rate: ${LR}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Max Steps: ${MAX_STEPS}"
echo "  TTC: ENABLED (Best-of-N with N=${TTC_INTERNAL_SAMPLES})"
echo "  Temperature: ${TEMPERATURE}"
echo ""

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run RLOO training
python src/main_rloo.py \
  --sft_model_path "${SFT_MODEL_PATH}" \
  --prompt_dataset_name "${DATASET_NAME}" \
  --prompt_dataset_split "train" \
  --output_dir "${OUTPUT_DIR}" \
  --k_samples ${K_SAMPLES} \
  --lr ${LR} \
  --batch_size ${BATCH_SIZE} \
  --max_steps ${MAX_STEPS} \
  --gradient_accumulation_steps ${GRAD_ACCUM_STEPS} \
  --max_length_generation ${MAX_LENGTH_GEN} \
  --temperature ${TEMPERATURE} \
  --ttc_internal_samples_n ${TTC_INTERNAL_SAMPLES} \
  --ttc_lambda_cost ${TTC_LAMBDA_COST} \
  --log_every_n_steps ${LOG_EVERY} \
  --save_every_n_steps ${SAVE_EVERY} \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "${WANDB_RUN_NAME}"

echo ""
echo "RLOO Optimized Training Complete!"
echo "Model saved to: ${OUTPUT_DIR}/final_model"
echo "To upload to Hugging Face, run: python upload_model.py --model_dir ${OUTPUT_DIR}/final_model" 