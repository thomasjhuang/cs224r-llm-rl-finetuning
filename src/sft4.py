#!/usr/bin/env python3
import os
import time
import math
import random
import argparse
import logging

import torch
from torch.utils.data import DataLoader, Sampler
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("sft_qwen2_bucketed")


class LengthBucketSampler(Sampler):
    """
    Wraps a dataset of known lengths, returning indices in “buckets” of size batch_size
    so that each mini‐batch has similar sequence lengths (reducing padding overhead).
    """
    def __init__(self, lengths, batch_size, shuffle=True):
        """
        lengths: list of sequence lengths (e.g. len(x["input_ids"]) for x in the dataset)
        batch_size: how many examples per batch
        shuffle: whether to shuffle buckets and within-bucket order
        """
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Sort all indices by ascending length
        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        # Partition sorted_indices into contiguous buckets of size batch_size
        self.buckets = [
            sorted_indices[i : i + batch_size]
            for i in range(0, len(sorted_indices), batch_size)
        ]

    def __iter__(self):
        buckets = self.buckets.copy()
        if self.shuffle:
            random.shuffle(buckets)
        for bucket in buckets:
            if self.shuffle:
                random.shuffle(bucket)
            yield from bucket

    def __len__(self):
        return len(self.lengths)


def collate_fn_pytorch(batch, pad_id):
    """
    Pads a list of examples (each with "input_ids" and "labels" tensors) to the maximum
    length within this batch. Returns CPU tensors only.
    """
    from torch.nn.utils.rnn import pad_sequence

    input_ids_list = [ex["input_ids"] for ex in batch]
    labels_list    = [ex["labels"]    for ex in batch]

    ids_padded  = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    labs_padded = pad_sequence(labels_list, batch_first=True, padding_value=-100)
    attn_mask   = (ids_padded != pad_id).long()

    return {
        "input_ids":      ids_padded,     # CPU tensor
        "labels":         labs_padded,     # CPU tensor
        "attention_mask": attn_mask,       # CPU tensor
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenized_data_dir", default="smoltalk_tokenized_900",
                   help="Path to the pre-tokenized dataset on disk")
    p.add_argument("--output_dir", default="runs/qwen2-sft-bucketed",
                   help="Directory where checkpoints and final model will be saved")
    p.add_argument("--epochs", type=int, default=3,
                   help="Number of training epochs")
    p.add_argument("--per_device_train_batch_size", type=int, default=4,
                   help="Batch size per GPU before gradient accumulation")
    p.add_argument("--gradient_accumulation_steps", type=int, default=4,
                   help="Number of steps to accumulate gradients")
    p.add_argument("--lr", type=float, default=2e-5,
                   help="Learning rate")
    p.add_argument("--max_length", type=int, default=900,
                   help="Maximum sequence length (used in preprocessing)")
    p.add_argument("--eval_every", type=int, default=20000,
                   help="Number of steps between evaluations (if implemented)")
    p.add_argument("--log_every", type=int, default=200,
                   help="Number of steps between logging training metrics")
    p.add_argument("--save_every", type=int, default=20000,
                   help="Number of steps between saving checkpoints")
    p.add_argument("--num_workers", type=int, default=8,
                   help="Number of DataLoader worker processes")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load pre‐tokenized dataset (with max_length=900)
    ds = load_from_disk(args.tokenized_data_dir)

    # 2) Precompute lengths for bucketing
    lengths = [len(x["input_ids"]) for x in ds]

    # 3) Create a length‐bucket sampler
    sampler = LengthBucketSampler(lengths, args.per_device_train_batch_size, shuffle=True)

    # 4) Load tokenizer & model
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True, use_fast=True)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model = torch.compile(model, backend="inductor")
    model.to(device).train()

    # 5) Optimizer + scheduler
    total = len(ds)
    steps_per_epoch = math.ceil(
        total / (args.per_device_train_batch_size * args.gradient_accumulation_steps)
    )
    max_steps = steps_per_epoch * args.epochs
    warmup_steps = int(max_steps * 0.01)

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, max_steps)

    # 6) DataLoader: bucketed + multi‐worker + pin_memory + prefetch
    ds.set_format(type="torch", columns=["input_ids", "labels"])
    loader = DataLoader(
        ds,
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=lambda b: collate_fn_pytorch(b, tok.pad_token_id),
    )

    from torch.amp import autocast

    run_loss, gstep, acc_steps = 0.0, 0, 0
    start_time = time.time()

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(loader):
            # Move batch to GPU on the main process
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # Use BF16 autocast (no GradScaler needed on H100)
            with autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
                loss = out.loss / args.gradient_accumulation_steps

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
                    elapsed = time.time() - start_time
                    speed = gstep / (elapsed + 1e-8)
                    avg_loss = run_loss / args.log_every
                    logger.info(f"step {gstep}/{max_steps} | avg-loss {avg_loss:.4f} | {speed:.2f} it/s")
                    run_loss = 0.0

                if gstep % args.save_every == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"step_{gstep}")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    model.save_pretrained(ckpt_dir)
                    tok.save_pretrained(ckpt_dir)

            if gstep >= max_steps:
                break
        if gstep >= max_steps:
            break

    logger.info("Training complete — saving final model")
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()