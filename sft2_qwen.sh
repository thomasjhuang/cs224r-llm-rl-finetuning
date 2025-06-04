#!/bin/bash
python src/sft2.py   \
--output_dir runs/qwen2-sft   \
--per_device_train_batch_size 4   \
--gradient_accumulation_steps 2   \
--epochs 3 \
--eval_every 5000 \
--log_every 100 \
--save_every 5000