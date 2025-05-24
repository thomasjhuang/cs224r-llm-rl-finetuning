#!/usr/bin/env python
from __future__ import annotations
import os, math, argparse, logging, time
from functools import partial

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, disable_caching
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

disable_caching()
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger("sft_qwen2")


def build_and_tokenize(sample, tokenizer, max_len):
    msgs = sample["messages"]
    idx = next((i for i in range(len(msgs) - 1, -1, -1) if msgs[i]["role"] == "assistant"), None)
    if idx is None:
        return {}
    prompt, completion = msgs[:idx], msgs[idx]
    prompt_ids = tokenizer(tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True),
                           add_special_tokens=False)["input_ids"]
    if len(prompt_ids) >= max_len - 1:
        return {}
    comp_ids = tokenizer(completion["content"].strip() + tokenizer.eos_token,
                         add_special_tokens=False, truncation=True,
                         max_length=max_len - len(prompt_ids))["input_ids"]
    return {"input_ids": prompt_ids + comp_ids, "labels": [-100]*len(prompt_ids)+comp_ids}

def collate(batch, tok):
    pad = tok.pad_token_id
    ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    labels = [torch.tensor(x["labels"], dtype=torch.long) for x in batch]
    ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return {"input_ids": ids, "labels": labels, "attention_mask": ids.ne(pad).long()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2-0.5B")
    p.add_argument("--dataset_name", default="HuggingFaceTB/smol-smoltalk")
    p.add_argument("--output_dir", default="./qwen2_sft")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.01)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval_pct", type=float, default=1.0, help="% of data for quick eval")
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=20000)
    p.add_argument("--wandb_project", default="qwen2-smoltalk-sft")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    try:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))
    except Exception:
        wandb = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    ).to(device).train()

    raw = load_dataset(args.dataset_name, split="train", streaming=args.streaming)
    proc = partial(build_and_tokenize, tokenizer=tok, max_len=args.max_length)
    train_ds = raw.map(proc, remove_columns=raw.features.keys()).filter(lambda x: "input_ids" in x)

    total = len(load_dataset(args.dataset_name, split="train", streaming=False))
    steps_per_epoch = math.ceil(total / (args.per_device_train_batch_size * args.gradient_accumulation_steps))
    max_steps = steps_per_epoch * args.epochs
    warmup = int(max_steps * args.warmup_ratio)

    loader = DataLoader(train_ds, batch_size=args.per_device_train_batch_size, collate_fn=lambda b: collate(b, tok))
    itr = iter(loader) if args.streaming else None

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9,0.95), weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(optim, warmup, max_steps)
    autocast_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if torch.cuda.is_available() else torch.amp.autocast('cuda')

    run_loss, gstep, asteps = 0.0, 0, 0; start=time.time()

    def log_train():
        speed = gstep / (time.time()-start+1e-8)
        avg = run_loss/args.log_every
        logger.info(f"step {gstep}/{max_steps} | avg‑loss {avg:.4f} | lr {sched.get_last_lr()[0]:.2e} | {speed:.2f} it/s")
        if wandb: wandb.log({"train_loss": avg, "lr": sched.get_last_lr()[0], "step": gstep})

    def evaluate():
        pct_str = ("%g" % args.eval_pct)
        try:
            raw_eval = load_dataset(args.dataset_name, split="test", streaming=True)
        except ValueError:
            n_take = int(total * args.eval_pct / 100)
            raw_eval = load_dataset(args.dataset_name, split="test", streaming=True).take(n_take)

        # Accumulate loss in plain Python loop to avoid map() writer issues
        total_loss, total_tok = 0.0, 0
        batch_buf = []

        def flush():
            nonlocal total_loss, total_tok, batch_buf
            if not batch_buf:
                return
            batch = collate(batch_buf, tok)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad(), autocast_ctx:
                out = model(**batch)
            n_tok = batch["labels"].ne(-100).sum().item()
            total_loss += out.loss.item() * n_tok
            total_tok += n_tok
            batch_buf = []

        for ex in raw_eval:
            ex_proc = proc(ex)
            if "input_ids" not in ex_proc:
                continue
            batch_buf.append(ex_proc)
            if len(batch_buf) == 4:
                flush()
        flush()

        ce = total_loss / max(total_tok, 1)
        ppl = math.exp(ce)
        logger.info(f"⚡ eval CE {ce:.4f} | ppl {ppl:.2f}")
        if wandb:
            wandb.log({"eval_ce": ce, "eval_ppl": ppl, "step": gstep})
        model.train()

    while gstep < max_steps:
        if args.streaming:
            try: batch = next(itr)
            except StopIteration: itr = iter(loader); batch = next(itr)
        else:
            if asteps==0:
                epoch_loader = DataLoader(train_ds, batch_size=args.per_device_train_batch_size, shuffle=True, collate_fn=lambda b: collate(b, tok))
                epoch_iter = iter(epoch_loader)
            try: batch = next(epoch_iter)
            except StopIteration: epoch_iter = iter(epoch_loader); batch = next(epoch_iter)
        with autocast_ctx:
            batch = {k:v.to(device) for k,v in batch.items()}
            loss = model(**batch).loss / args.gradient_accumulation_steps
        run_loss += loss.item(); loss.backward(); asteps += 1
        if asteps == args.gradient_accumulation_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optim.step(); sched.step(); optim.zero_grad(); gstep+=1; asteps=0
            if gstep % args.log_every==0: log_train(); run_loss=0.0
            if gstep % args.eval_every==0: evaluate()
            if gstep % args.save_every==0:
                ck=os.path.join(args.output_dir,f"step_{gstep}"); model.save_pretrained(ck); tok.save_pretrained(ck)
    logger.info("Training finished – saving final…"); model.save_pretrained(args.output_dir); tok.save_pretrained(args.output_dir)
    if wandb: wandb.finish()

if __name__=="__main__":
    main()
