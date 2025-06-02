#!/usr/bin/env python
from __future__ import annotations
import os, math, argparse, logging, time
from functools import partial
import wandb
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, disable_caching
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from src.methods.dpo import DPOMethod
from src.data_handling.datasets import DPODataset
from src.utils.data_utils import DataCollatorForDPO

disable_caching()
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger("dpo_qwen2")

def get_device_and_dtype():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
        use_amp = False
        logger.info("Using CUDA with bfloat16")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32
        use_amp = False
        logger.info("Using MPS with float32 (no mixed precision)")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        use_amp = False
        logger.info("Using CPU with float32 (no mixed precision)")
    
    return device, dtype, use_amp

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="Path to SFT model")
    p.add_argument("--dataset_name", default="HuggingFaceH4/ultrafeedback_binarized")
    p.add_argument("--output_dir", default="./qwen2_dpo")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--wandb_project", default="qwen2-ultrafeedback-dpo")
    p.add_argument("--subset", type=int, default=100)
    return p.parse_args()

def main():
    args = parse_args()
    logger.info(f"Effective runtime arguments: {vars(args)}")
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    wandb.init(project=args.wandb_project, config=vars(args))

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device, dtype, use_amp = get_device_and_dtype()
    
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device).train()

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device).eval()
    
    for param in ref_model.parameters():
        param.requires_grad = False

    dpo_method = DPOMethod(policy=model, reference_model=ref_model, beta=args.beta, device=device)

    train_dataset = DPODataset(
        dataset_name=args.dataset_name,
        tokenizer=tok,
        split="train_prefs",
        max_length=args.max_length,
        subset=args.subset
    )

    collator = DataCollatorForDPO(tokenizer=tok, max_length=args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator
    )

    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    model.train()
    global_step = 0
    running_loss = 0.0
    running_metrics = {}
    accumulation_steps = 0
    start_time = time.time()

    logger.info(f"Starting training with {len(train_loader)} batches per epoch, {total_steps} total steps")

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            try:
                loss, metrics = dpo_method.compute_loss(batch)
                loss = loss / args.gradient_accumulation_steps
                
                if use_amp and scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                running_loss += loss.item()
                accumulation_steps += 1
                
                # Accumulate metrics
                for key, value in metrics.items():
                    if key not in running_metrics:
                        running_metrics[key] = 0.0
                    running_metrics[key] += value / args.gradient_accumulation_steps

                if accumulation_steps == args.gradient_accumulation_steps:
                    if use_amp and scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    accumulation_steps = 0

                    if global_step % args.log_every == 0:
                        avg_loss = running_loss / args.log_every
                        speed = global_step / (time.time() - start_time + 1e-8)
                        
                        # Average the metrics
                        avg_metrics = {}
                        for key, value in running_metrics.items():
                            avg_metrics[key] = value / args.log_every
                        
                        logger.info(f"Step {global_step}/{total_steps} | Loss {avg_loss:.4f} | " +
                                  f"RewAcc {avg_metrics.get('reward_accuracies', 0):.3f} | " +
                                  f"LR {scheduler.get_last_lr()[0]:.2e} | {speed:.2f} it/s")
                        logger.info(f"  P_ChosenLogP: {avg_metrics.get('policy_chosen_logp', 0):.2f}, P_RejectedLogP: {avg_metrics.get('policy_rejected_logp', 0):.2f}")
                        logger.info(f"  R_ChosenLogP: {avg_metrics.get('ref_chosen_logp', 0):.2f}, R_RejectedLogP: {avg_metrics.get('ref_rejected_logp', 0):.2f}")
                        logger.info(f"  LogitsStd: {avg_metrics.get('logits_std', 0):.3f}")
                        logger.info(f"  ChosenRewards: {avg_metrics.get('chosen_rewards_mean', 0):.3f}, RejectedRewards: {avg_metrics.get('rejected_rewards_mean', 0):.3f}")
                        logger.info(f"  RewardMargins: {avg_metrics.get('reward_margins_mean', 0):.3f}")
                        
                        # Log to wandb
                        wandb_log = {"train_loss": avg_loss, "lr": scheduler.get_last_lr()[0], "step": global_step}
                        wandb_log.update(avg_metrics)
                        wandb.log(wandb_log)
                        
                        running_loss = 0.0
                        running_metrics = {}

                    if global_step % args.save_every == 0:
                        save_path = os.path.join(args.output_dir, f"step_{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        model.save_pretrained(save_path)
                        tok.save_pretrained(save_path)

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"Out of memory at step {global_step}. Try reducing batch size or max_length.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    break
                else:
                    raise e

    logger.info("Training finished - saving final model")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    wandb.finish()

if __name__ == "__main__":
    main()
