#!/bin/bash
python src/sft_qwen_torch.py   \
--model_name runs/qwen2-sft/step_20000/ \
--output_dir runs/qwen2-sft-2   \
--per_device_train_batch_size 22   \
--gradient_accumulation_steps 3   \
--epochs 50 \
--eval_every 500 \
--log_every 100 \
--save_every 2000
