#!/usr/bin/env python
import os, time, math, argparse, logging
import torch
import wandb
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("sft_qwen2_fast")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2-0.5B")
    p.add_argument("--tokenized_data_dir", default="smoltalk_tokenized")
    p.add_argument("--output_dir", default="runs/qwen2-sft")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.01)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=10000)
    p.add_argument("--wandb_project", default="qwen2-smoltalk-sft")
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()

def collate(batch, tok):
    pad_id = tok.pad_token_id
    ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    labels = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]
    ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    attention_mask = (ids != pad_id).long()
    return {"input_ids": ids, "labels": labels, "attention_mask": attention_mask}

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    wandb.init(project=args.wandb_project, config=vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    ).to(device).train()

    # Load pre-tokenized dataset
    train_ds = load_from_disk(args.tokenized_data_dir)
    total = len(train_ds)
    steps_per_epoch = math.ceil(total / (args.per_device_train_batch_size * args.gradient_accumulation_steps))
    max_steps = steps_per_epoch * args.epochs
    warmup_steps = int(max_steps * args.warmup_ratio)

    loader = DataLoader(
        train_ds,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate(b, tok),
        prefetch_factor=2,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, max_steps)

    # Mixed-precision (bf16) on CUDA
    if torch.cuda.is_available():
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        import contextlib
        autocast_ctx = contextlib.nullcontext()

    run_loss, gstep, acc_steps = 0.0, 0, 0
    start_time = time.time()

    def log_train():
        elapsed = time.time() - start_time
        speed = gstep / (elapsed + 1e-8)
        avg_loss = run_loss / args.log_every
        lr_now = sched.get_last_lr()[0]
        logger.info(f"step {gstep}/{max_steps} | avg-loss {avg_loss:.4f} | lr {lr_now:.2e} | {speed:.2f} it/s")
        wandb.log({"train_loss": avg_loss, "lr": lr_now, "step": gstep})

    # Create a single iterator for the entire run (reshuffle each epoch automatically)
    train_iter = iter(loader)

    while gstep < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            # End of epoch: re-create iterator
            train_iter = iter(loader)
            batch = next(train_iter)

        with autocast_ctx:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss = model(**batch).loss / args.gradient_accumulation_steps

        run_loss += loss.item()
        loss.backward()
        acc_steps += 1

        if acc_steps == args.gradient_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            gstep += 1
            acc_steps = 0

            if gstep % args.log_every == 0:
                log_train()
                run_loss = 0.0

            if gstep % args.eval_every == 0:
                # (You can insert your `evaluate()` here.)
                pass

            if gstep % args.save_every == 0:
                ckpt_dir = os.path.join(args.output_dir, f"step_{gstep}")
                os.makedirs(ckpt_dir, exist_ok=True)
                model.save_pretrained(ckpt_dir)
                tok.save_pretrained(ckpt_dir)

    # Final save
    logger.info("Training finished — saving final model.")
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    wandb.finish()

if __name__ == "__main__":
    main()