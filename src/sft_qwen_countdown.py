#!/usr/bin/env python
import os, math, argparse, logging, time
from functools import partial
import wandb
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, disable_caching, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup, TrainingArguments, Trainer, DataCollatorForLanguageModeling, IntervalStrategy, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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
    
    tokenized_output = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        max_length=max_len,
        truncation=True,
        padding="max_length",  # Pad to max_len to ensure all tensors are the same size for distributed training
    )

    labels = list(tokenized_output)

    if len(labels) > max_len:
        return None

    return {
        "input_ids": labels, 
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
    # Core model and data
    p.add_argument("--model_name", default="Qwen/Qwen2-0.5B")
    p.add_argument("--dataset_name", default="Asap7772/cog_behav_all_strategies")
    p.add_argument("--output_dir", default="./qwen2_warmstart_sft_trainer")
    
    # Training parameters
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=1024)
    
    # Logging and saving
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=5)
    
    # W&B
    p.add_argument("--wandb_project", default="qwen2-warmstart-sft")
    p.add_argument("--wandb_run_name", default="Qwen2-0.5B-warmstart-sft")
    
    # Optional
    p.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to reduce memory usage")
    
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    logger.info(f"Loading dataset: {args.dataset_name}")
    
    # Load training dataset
    raw_train_dataset = load_dataset(args.dataset_name, split="train", streaming=False)

    if len(raw_train_dataset) > 0:
        sample = raw_train_dataset[0]
        logger.info(f"Sample train dataset fields: {list(sample.keys())}")

    proc_fn = partial(build_and_tokenize_countdown, tokenizer=tok, max_len=args.max_length)
    
    num_proc = min(os.cpu_count() // 2 if os.cpu_count() else 1, 8)

    logger.info(f"Mapping train dataset...")
    train_dataset = raw_train_dataset.map(
        proc_fn, 
        batched=False, 
        num_proc=num_proc,
        remove_columns=raw_train_dataset.column_names
    ).filter(lambda x: x is not None and "input_ids" in x and len(x["input_ids"]) > 0)
    
    if isinstance(train_dataset, Dataset):
        train_dataset = train_dataset.shuffle(seed=42)

    logger.info(f"Processed train dataset samples: {len(train_dataset)}")

    data_collator = partial(collate, tok=tok)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="wandb",
        run_name=args.wandb_run_name,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False),
        tokenizer=tok,
    )

    logger.info("Starting training...")
    
    # Check for existing checkpoints and resume if available
    resume_from_checkpoint = None
    if os.path.exists(args.output_dir):
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            # Sort by checkpoint number and get the latest
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
            resume_from_checkpoint = os.path.join(args.output_dir, latest_checkpoint)
            logger.info(f"Resuming training from {resume_from_checkpoint}")
    
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    logger.info("Training finished. Saving...")
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    
    if wandb.run:
        wandb.finish()

if __name__ == "__main__":
    main() 