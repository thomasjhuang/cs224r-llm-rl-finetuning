#!/bin/bash

cd ..
python -m src.main_dpo \
    --model_path "anatal/qwen2_05_smol-smoltalk" \
    --dataset_name "HuggingFaceH4/ultrafeedback_binarized" \
    --output_dir "./qwen2_dpo/run_$(date +%Y%m%d_%H%M%S)" \
    --epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr 5e-7 \
    --warmup_ratio 0.1 \
    --max_length 512 \
    --beta 0.3 \
    --subset 15000 \
    --log_every 10 \
    --save_every 200 \
    --wandb_project "qwen2-dpo-stable" 