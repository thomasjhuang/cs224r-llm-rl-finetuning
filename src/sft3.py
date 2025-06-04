import os, time, math, argparse, logging
import torch
from torch.utils.data import DataLoader, Sampler
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)
logger = logging.getLogger("sft_qwen2_peak")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenized_data_dir", default="data/smoltalk_tokenized")
    p.add_argument("--output_dir", default="runs/qwen2-sft")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=800)  # cut down from 1024
    p.add_argument("--eval_every", type=int, default=20000)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--save_every", type=int, default=20000)
    p.add_argument("--num_workers", type=int, default=8)
    return p.parse_args()

def collate_fn_pytorch(batch, pad_id):
    from torch.nn.utils.rnn import pad_sequence
    ids = [ex["input_ids"] for ex in batch]
    lbl = [ex["labels"]    for ex in batch]
    ids_padded = pad_sequence(ids, batch_first=True, padding_value=pad_id)
    lbl_padded = pad_sequence(lbl, batch_first=True, padding_value=-100)
    attn_mask = (ids_padded != pad_id).long()
    return {
        "input_ids": ids_padded,
        "labels":    lbl_padded,
        "attention_mask": attn_mask,
    }

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset (already tokenized with max_length=800)
    train_ds = load_from_disk(args.tokenized_data_dir)
    total = len(train_ds)
    steps_per_epoch = math.ceil(total / (args.per_device_train_batch_size * args.gradient_accumulation_steps))
    max_steps = steps_per_epoch * args.epochs
    warmup_steps = int(max_steps * 0.01)

    # Load tokenizer & model (FP16)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True, use_fast=True)
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-0.5B",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()             # still helpful if you bump batch to 6–8
    model = torch.compile(model, backend="inductor")  # fusion
    model.to(device).train()

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, max_steps)

    # DataLoader
    train_ds.set_format(type="torch", columns=["input_ids", "labels"])
    loader = DataLoader(
        train_ds,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=lambda b: collate_fn_pytorch(b, tok.pad_token_id),
    )

    from torch.amp import autocast, GradScaler
    scaler = GradScaler()

    run_loss, gstep, acc_steps = 0.0, 0, 0
    start_time = time.time()

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(loader):
            # Move batch to GPU on the main process
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with autocast("cuda", dtype=torch.float16):
                out = model(**batch)
                loss = out.loss / args.gradient_accumulation_steps

            run_loss += loss.item()
            scaler.scale(loss).backward()
            acc_steps += 1

            if acc_steps == args.gradient_accumulation_steps:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
                sched.step()
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
