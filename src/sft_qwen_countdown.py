#!/usr/bin/env python
import os, math, argparse, logging, time
from functools import partial
import wandb
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, disable_caching, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup, TrainingArguments, Trainer, DataCollatorForLanguageModeling, IntervalStrategy

disable_caching()
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
logger = logging.getLogger("sft_qwen2_countdown")

def build_and_tokenize_countdown(sample, tokenizer, max_len):
    problem, solution = None, None

    if hasattr(build_and_tokenize_countdown, 'call_count'):
        build_and_tokenize_countdown.call_count += 1
    else:
        build_and_tokenize_countdown.call_count = 1
    
    if build_and_tokenize_countdown.call_count <= 3:
        logger.info(f"Sample {build_and_tokenize_countdown.call_count} keys: {list(sample.keys())}")

    if 'query' in sample and 'completion' in sample:
        problem = sample['query']
        solution = sample['completion']
    elif 'problem' in sample and 'solution' in sample:
        problem = sample['problem']
        solution = sample['solution']
    elif 'input' in sample and 'output' in sample:
        problem = sample['input']
        solution = sample['output']
    elif 'question' in sample and 'answer' in sample:
        problem = sample['question']
        solution = sample['answer']
    elif 'text' in sample:
        text = sample['text']
        if '### Problem:' in text and '### Solution:' in text:
            parts = text.split('### Solution:', 1)
            problem_part = parts[0].replace('### Problem:', '').strip()
            solution_part = parts[1].strip()
            problem = problem_part
            solution = solution_part
        elif '###' in text:
            parts = text.split('###', 1)
            if len(parts) >= 2:
                problem = parts[0].strip()
                solution = parts[1].strip()
    
    if problem is None or solution is None or len(problem.strip()) == 0 or len(solution.strip()) == 0:
        return None

    messages = [
        {"role": "user", "content": problem.strip()},
        {"role": "assistant", "content": solution.strip()}
    ]
    
    prompt_text = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, truncation=True, max_length=max_len//2)["input_ids"]

    if len(prompt_ids) >= max_len - 10:
        return None
    
    completion_text = messages[1]["content"].strip() + tokenizer.eos_token
    comp_ids = tokenizer(completion_text, 
                         add_special_tokens=False, 
                         truncation=True,
                         max_length=max_len - len(prompt_ids))["input_ids"]
    
    input_ids = prompt_ids + comp_ids
    labels = [-100] * len(prompt_ids) + comp_ids

    if len(input_ids) > max_len:
        return None

    return {
        "input_ids": input_ids, 
        "labels": labels
    }

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
    p.add_argument("--dataset_name", default="Asap7772/cog_behav_all_strategies")
    p.add_argument("--output_dir", default="./qwen2_warmstart_sft_trainer")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.01)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval_pct", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--save_every", type=int, default=10000)
    p.add_argument("--wandb_project", default="qwen2-warmstart-sft")
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--eval_dataset_split", default="test")
    p.add_argument("--max_eval_samples", type=int, default=500)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--save_total_limit", type=int, default=2)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    wandb.init(project=args.wandb_project, config=vars(args), name=f"{args.model_name.split('/')[-1]}-{args.dataset_name.split('/')[-1].replace('cog_behav_all_strategies', 'warmstart')}-sft")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    logger.info(f"Loading dataset: {args.dataset_name}")
    
    is_streaming = args.streaming
    raw_train_dataset = load_dataset(args.dataset_name, split=args.dataset_split, streaming=is_streaming)
    raw_eval_dataset = load_dataset(args.dataset_name, split=args.eval_dataset_split, streaming=False) 
    if args.max_eval_samples and len(raw_eval_dataset) > args.max_eval_samples:
         raw_eval_dataset = raw_eval_dataset.select(range(args.max_eval_samples))

    if is_streaming:
        logger.info("Using streaming for train dataset")
        sample = next(iter(raw_train_dataset))
        logger.info(f"Sample train dataset fields: {list(sample.keys())}")
    else:
        if len(raw_train_dataset) > 0:
            sample = raw_train_dataset[0]
            logger.info(f"Sample train dataset fields: {list(sample.keys())}")

    proc_fn = partial(build_and_tokenize_countdown, tokenizer=tok, max_len=args.max_length)
    
    num_proc = min(os.cpu_count() // 2 if os.cpu_count() else 1, 8) if not is_streaming else None

    logger.info(f"Mapping train dataset...")
    train_dataset = raw_train_dataset.map(
        proc_fn, 
        batched=False, 
        remove_columns=raw_train_dataset.column_names if not is_streaming else None
    ).filter(lambda x: x is not None and "input_ids" in x and len(x["input_ids"]) > 0)
    
    logger.info("Mapping eval dataset...")
    eval_dataset = raw_eval_dataset.map(
        proc_fn, 
        batched=False, 
        num_proc=num_proc, 
        remove_columns=raw_eval_dataset.column_names
    ).filter(lambda x: x is not None and "input_ids" in x and len(x["input_ids"]) > 0)

    if is_streaming:
        train_dataset = train_dataset.map(lambda x: {"input_ids": x["input_ids"], "labels": x["labels"]}, batched=False)
        
    if not is_streaming and isinstance(train_dataset, Dataset):
        train_dataset = train_dataset.shuffle(seed=42)

    logger.info(f"Processed train dataset samples: {len(train_dataset) if not is_streaming else 'streaming'}")
    logger.info(f"Processed eval dataset samples: {len(eval_dataset)}")

    data_collator = partial(collate, tok=tok)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        logging_dir=f"{args.output_dir}/logs",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=False,
        report_to="wandb",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    def compute_metrics(eval_preds):
        return {"eval_loss": 0.0}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tok,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    train_result = trainer.train()
    
    logger.info("Training finished. Saving...")
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    logger.info("Evaluating...")
    eval_metrics = trainer.evaluate()
    logger.info(f"Final evaluation metrics: {eval_metrics}")
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    if wandb.run:
        wandb.finish()

if __name__ == "__main__":
    main() 