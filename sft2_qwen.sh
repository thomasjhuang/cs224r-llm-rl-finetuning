#!/bin/bash
python src/sft4.py   \
--tokenized_data_dir data/smoltalk_tokenized \
--output_dir runs/qwen2-sft   \
--per_device_train_batch_size 16   \
--gradient_accumulation_steps 2   \
--epochs 300 \
--eval_every 5000 \
--log_every 100 \
--save_every 5000
