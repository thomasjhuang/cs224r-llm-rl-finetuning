#!/bin/bash
python src/sft_qwen_torch.py   \
--output_dir runs/qwen2-sft   \
--per_device_train_batch_size 22   \
--gradient_accumulation_steps 3   \
--epochs 300 \
--eval_every 5000 \
--log_every 100 \
--save_every 10000
