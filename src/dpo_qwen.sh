#!/bin/bash

cd ..
python -m src.main_dpo \
    --model_path "anatal/qwen2_05_smol-smoltalk" \
    --dataset_name "HuggingFaceH4/ultrafeedback_binarized" \
    --output_dir "./qwen2_dpo" \
    --epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --lr 5e-7 \
    --warmup_ratio 0.1 \
    --max_length 256 \
    --beta 0.1 \
    --subset 50 \
    --log_every 5 \
    --save_every 25 \
    --wandb_project "qwen2-ultrafeedback-dpo" 