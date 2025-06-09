#!/bin/bash

# RLOO Base Experiment - Vanilla RLOO without TTC
# CS224R Final Project - RL Fine-tuning of Language Models

set -e  # Exit on any error

# Activate conda environment
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate cs224r-project

# Navigate to project directory
cd /home/ubuntu/cs224r-project/cs224r-llm-rl-finetuning

echo "Starting RLOO Base Experiment (Vanilla RLOO without TTC)"
echo "============================================================"

# Check if SFT warmstart checkpoint exists
if [ ! -d "./qwen2_warmstart_sft_trainer/checkpoint-1000" ]; then
    echo "Error: SFT warmstart checkpoint not found at ./qwen2_warmstart_sft_trainer/checkpoint-1000"
    echo "Please run SFT training first using: bash sft_warmup.sh"
    exit 1
fi

# RLOO Base Parameters
SFT_MODEL_PATH="./qwen2_warmstart_sft_trainer/checkpoint-1000"
DATASET_NAME="Jiayi-Pan/Countdown-Tasks-3to4"
OUTPUT_DIR="./rloo_base_qwen_trainer"
WANDB_PROJECT="qwen2-countdown-rloo"
WANDB_RUN_NAME="rloo-base-vanilla"

# Training hyperparameters
K_SAMPLES=4
LR=5e-6
BATCH_SIZE=2
MAX_STEPS=50
GRAD_ACCUM_STEPS=4
MAX_LENGTH_GEN=100
TEMPERATURE=0.7

# TTC disabled (vanilla RLOO)
TTC_INTERNAL_SAMPLES=1
TTC_LAMBDA_COST=0.0

# Logging
LOG_EVERY=1
SAVE_EVERY=25

echo "Configuration:"
echo "  Model: ${SFT_MODEL_PATH}"
echo "  Dataset: ${DATASET_NAME}"
echo "  Output: ${OUTPUT_DIR}"
echo "  k_samples: ${K_SAMPLES}"
echo "  Learning Rate: ${LR}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  Max Steps: ${MAX_STEPS}"
echo "  TTC: Disabled (vanilla RLOO)"
echo ""

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Run RLOO training
python src/main_rloo.py \
  --sft_model_path "${SFT_MODEL_PATH}" \
  --prompt_dataset_name "${DATASET_NAME}" \
  --prompt_dataset_split "train" \
  --max_prompt_samples 1000 \
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
echo "RLOO Base Training Complete!"
echo "Model saved to: ${OUTPUT_DIR}/final_model"
echo "To upload to Hugging Face, run: python upload_model.py --model_dir ${OUTPUT_DIR}/final_model" 