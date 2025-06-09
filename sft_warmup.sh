#!/bin/bash

# Set environment variable to avoid W&B issues in loops/scripts
export WANDB_START_METHOD="thread"

echo "Starting distributed SFT training..."
echo "Will resume automatically from latest checkpoint if available..."

torchrun --nproc_per_node=8 src/sft_qwen_countdown.py \
  --model_name "Qwen/Qwen2-0.5B" \
  --dataset_name "Asap7772/cog_behav_all_strategies" \
  --output_dir "./qwen2_warmstart_sft_trainer" \
  --max_steps 1500 \
  --save_steps 100 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --lr 2e-5 \
  --logging_steps 50 \
  --save_total_limit 5 \
  --wandb_run_name "Qwen2-0.5B-SFT-resume-no-eval" \
  --gradient_checkpointing

echo "Training completed!"

# Check if we got checkpoint-1000
if [ -d "./qwen2_warmstart_sft_trainer/checkpoint-1000" ]; then
    echo "✅ SUCCESS: checkpoint-1000 created!"
    echo "Ready for RLOO training!"
else
    HIGHEST=$(ls -d ./qwen2_warmstart_sft_trainer/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1 || echo "0")
    echo "⚠️  Highest checkpoint: checkpoint-${HIGHEST}"
fi 