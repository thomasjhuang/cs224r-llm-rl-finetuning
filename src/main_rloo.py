# RLOO training script

import argparse
import logging
import os
import torch
import torch.nn as nn
import wandb
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import json

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.methods.rloo import RLOOTrainer

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using RLOO on Countdown task.")
    parser.add_argument("--sft_model_path", type=str, default="./qwen2_warmstart_sft_trainer/checkpoint-1000", help="Path to the SFT warmstarted model checkpoint.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the tokenizer, defaults to sft_model_path if None.")
    parser.add_argument("--prompt_dataset_name", type=str, default="Jiayi-Pan/Countdown-Tasks-3to4", help="Dataset for prompts.")
    parser.add_argument("--prompt_dataset_split", type=str, default="train", help="Split of the prompt dataset to use.")
    parser.add_argument("--max_prompt_samples", type=int, default=None, help="Maximum number of prompt samples to use (for quick testing).")
    parser.add_argument("--output_dir", type=str, default="./rloo_qwen_countdown_trainer", help="Directory to save RLOO model checkpoints.")
    
    parser.add_argument("--k_samples", type=int, default=4, help="Number of samples per prompt for RLOO.")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate for RLOO.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for prompts.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum training steps (overrides num_epochs if set).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--max_length_generation", type=int, default=100, help="Max new tokens for generation.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling during generation.")
    parser.add_argument("--log_every_n_steps", type=int, default=10, help="Log metrics every N steps.")
    parser.add_argument("--save_every_n_steps", type=int, default=200, help="Save checkpoint every N steps.")
    parser.add_argument("--wandb_project", type=str, default="qwen2-countdown-rloo", help="WandB project name.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name.")
    
    # TTC related arguments
    parser.add_argument("--ttc_internal_samples_n", type=int, default=1, help="Number of internal samples/drafts to generate for each RLOO sample. Cost is based on this.")
    parser.add_argument("--ttc_lambda_cost", type=float, default=0.0, help="Lambda penalty coefficient for TTC computational cost (applied per internal sample).")
    
    return parser.parse_args()

def prepare_prompt_for_generation(sample, tokenizer, use_cot=True):
    numbers_str = ", ".join(map(str, sample['nums']))
    
    if use_cot:
        # Chain-of-Thought prompt for TTC
        problem_text = f"Using the numbers [{numbers_str}], create an equation that equals {sample['target']}. Think step by step and show your reasoning."
    else:
        # Direct prompt (vanilla)
        problem_text = f"Using the numbers [{numbers_str}], create an equation that equals {sample['target']}."
    
    messages = [{"role": "user", "content": problem_text}]
    prompt_chat_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return {"prompt_text": prompt_chat_template, "original_data": sample}

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    wandb_run_name = args.wandb_run_name if args.wandb_run_name else f"rloo-k{args.k_samples}-lr{args.lr}-bs{args.batch_size}"
    wandb.init(project=args.wandb_project, name=wandb_run_name, config=vars(args))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        logger.info(f"Using device: {device}")
        if num_gpus > 1:
            logger.info(f"Using {num_gpus} GPUs for training.")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU.")

    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.sft_model_path
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Loading SFT model from: {args.sft_model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.sft_model_path, torch_dtype=torch.float32)
    model.to(device)

    if torch.cuda.device_count() > 1:
        logger.info(f"Applying torch.nn.DataParallel to the model for {torch.cuda.device_count()} GPUs.")
        model = nn.DataParallel(model)

    logger.info(f"Loading prompt dataset: {args.prompt_dataset_name}, split: {args.prompt_dataset_split}")
    prompt_dataset = load_dataset(args.prompt_dataset_name, split=args.prompt_dataset_split)
    if args.max_prompt_samples:
        prompt_dataset = prompt_dataset.select(range(min(len(prompt_dataset), args.max_prompt_samples)))
    
    def collate_fn(batch):
        prompt_texts = [item['prompt_text'] for item in batch]
        original_datas = [item['original_data'] for item in batch]
        
        tokenized_prompts = tokenizer(prompt_texts, return_tensors='pt', padding=True, truncation=True, max_length=tokenizer.model_max_length // 2).to(device)
        return tokenized_prompts['input_ids'], tokenized_prompts['attention_mask'], prompt_texts, original_datas

    # Use Chain-of-Thought prompting when TTC is enabled
    use_cot = args.ttc_internal_samples_n > 1 and args.ttc_lambda_cost > 0
    processed_prompt_dataset = prompt_dataset.map(lambda x: prepare_prompt_for_generation(x, tokenizer, use_cot), batched=False)
    
    prompt_dataloader = DataLoader(processed_prompt_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = (len(prompt_dataloader) // args.gradient_accumulation_steps) * args.num_epochs
    if args.max_steps:
        num_training_steps = args.max_steps
    
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    rloo_trainer = RLOOTrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        k_samples=args.k_samples,
        max_length_generation=args.max_length_generation,
        temperature=args.temperature,
        device=device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ttc_internal_samples_n=args.ttc_internal_samples_n,
        ttc_lambda_cost=args.ttc_lambda_cost
    )

    logger.info("Starting RLOO training...")
    total_optimizer_steps_taken = 0 

    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        
        if args.max_steps:
            remaining_optimizer_steps = args.max_steps - total_optimizer_steps_taken
            remaining_micro_batches = remaining_optimizer_steps * args.gradient_accumulation_steps
            current_loop_total = min(len(prompt_dataloader), remaining_micro_batches if remaining_optimizer_steps > 0 else 0)
        else:
            current_loop_total = len(prompt_dataloader)

        progress_bar = tqdm(prompt_dataloader, desc=f"Epoch {epoch+1}", total=current_loop_total, disable=current_loop_total==0)
        
        for batch_idx, (batch_input_ids, batch_attention_mask, batch_prompt_texts, batch_original_data) in enumerate(progress_bar):
            
            if args.max_steps and total_optimizer_steps_taken >= args.max_steps:
                break

            metrics = rloo_trainer.train_step(batch_input_ids, batch_attention_mask, batch_prompt_texts, batch_original_data)
            
            if rloo_trainer.grad_accum_count == 0:
                total_optimizer_steps_taken += 1
                progress_bar.set_postfix(loss=metrics['loss'], avg_reward=metrics['avg_reward'], opt_step=total_optimizer_steps_taken)

                if total_optimizer_steps_taken % args.log_every_n_steps == 0:
                    wandb.log({"epoch": epoch + 1, "optimizer_step": total_optimizer_steps_taken, **metrics})
                    
                    # Construct log message for console
                    log_message_parts = [
                        f"Optimizer Step {total_optimizer_steps_taken}",
                        f"Loss={metrics['loss']:.4f}",
                        f"AvgReward={metrics['avg_reward']:.4f}" # This is the TTC-adjusted reward
                    ]
                    if 'avg_actual_reward' in metrics:
                        log_message_parts.append(f"AvgActualReward={metrics['avg_actual_reward']:.4f}") # Original Countdown reward
                    if 'avg_tokens_generated' in metrics: # Assuming this might be added later
                        log_message_parts.append(f"AvgTokensGen={metrics['avg_tokens_generated']:.2f}")
                    if 'avg_internal_samples_used' in metrics: # For new TTC logging
                        log_message_parts.append(f"AvgInternalSamples={metrics['avg_internal_samples_used']:.2f}")
                    
                    logger.info(", ".join(log_message_parts))
            
                if args.save_every_n_steps > 0 and total_optimizer_steps_taken % args.save_every_n_steps == 0:
                    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-optimizer-{total_optimizer_steps_taken}")
                    rloo_trainer.save_model(checkpoint_dir)
        
        if args.max_steps and total_optimizer_steps_taken >= args.max_steps:
            logger.info(f"Reached max_optimizer_steps ({args.max_steps}). Stopping training.")
            break

    logger.info("Training finished. Saving final model.")
    final_model_dir = os.path.join(args.output_dir, "final_model")
    rloo_trainer.save_model(final_model_dir)
    
    with open(os.path.join(args.output_dir, "training_args.json"), 'w') as f:
        json.dump(vars(args), f, indent=2)

    wandb.finish()
    logger.info("RLOO training complete.")

if __name__ == "__main__":
    main()
