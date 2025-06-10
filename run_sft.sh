#!/bin/bash

# SFT Warmup Training Script
# This script fine-tunes the base Qwen2 model on the general-purpose
# Asap7772/cog_behav_all_strategies dataset to create a robust
# instruction-following model before RL.

set -e # Exit on any error

# Activate conda environment
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate cs224r-project

echo "Starting SFT Warmup Training with DistributedDataParallel..."
echo "=============================================================="


torchrun --nproc_per_node=8 src/sft_qwen_warmup.py \
    --model_name "Qwen/Qwen2-0.5B" \
    --dataset_name "Asap7772/cog_behav_all_strategies" \
    --output_dir "./sft_warmup_model" \
    --max_steps 1000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --save_steps 100 \
    --logging_steps 10 \
    --lr 2e-5 \
    --wandb_project "qwen2-warmup-sft" \
    --wandb_run_name "sft-warmup-asap7772-countdown-filtered"

echo "SFT warmup training complete. Model saved in ./sft_warmup_model" 