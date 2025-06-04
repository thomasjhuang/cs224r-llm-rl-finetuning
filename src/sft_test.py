import torch
import time
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from multiprocessing import cpu_count

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_NAME        = "Qwen/Qwen2.5-0.5B"
SEQ_LEN           = 1024
BATCH_SIZE        = 64
NUM_WARMUP_STEPS  = 10
NUM_MEASURE_STEPS = 50
DEVICE            = "cuda"
LEARNING_RATE     = 1e-4
PROJECT_NAME      = "qwen2-smoltalk-sft-test"
NUM_WORKERS       = 50

# ─── INITIALIZE WANDB ─────────────────────────────────────────────────────────
wandb.init(
    project=PROJECT_NAME,
    config={
        "model_name": MODEL_NAME,
        "seq_len": SEQ_LEN,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_warmup_steps": NUM_WARMUP_STEPS,
        "num_measure_steps": NUM_MEASURE_STEPS,
        "device": DEVICE,
    }
)
config = wandb.config

# ─── LOAD TOKENIZER + MODEL ───────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Ensure a pad_token exists (many causal‐LMs don’t define one by default).
# Here we simply use eos_token as pad_token if pad_token_id is missing.
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to(DEVICE)
model.train()

# ─── LOAD & INSPECT SMOL-SMOLTALK ──────────────────────────────────────────────
raw_ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")

# Print column names so you can see which keys are present (e.g. “text”, “prompt”, etc.)
print(f"Dataset columns: {raw_ds.column_names}\n")
print("Example #0 of train-split:\n", raw_ds[0], "\n")
print("-----------\nIf you see neither 'text' nor 'prompt'/'completion' nor 'instruction'/'response',\n"
      "then adjust the branches in `collate_fn` below to match your actual keys.\n")

# Split 90% train / 10% validation
split_ds = raw_ds.train_test_split(test_size=0.10, seed=42)
train_ds = split_ds["train"]
eval_ds  = split_ds["test"]

# ─── COLLATE FUNCTION ──────────────────────────────────────────────────────────
def collate_fn(batch):
    """
    Tries, in order:
      1) ex["text"]
      2) ex["prompt"] + ex["completion"]
      3) ex["instruction"] + ex["response"]
    and then tokenizes that concatenated string.

    Finally:
      - Pads/truncates to SEQ_LEN,
      - Builds labels = input_ids but masks out pad positions (label = -100 on pad).
    """
    texts = []
    for ex in batch:
        # 1) If your dataset has a single "text" field:
        if "text" in ex:
            txt = ex["text"]

        # 2) If your dataset has separate "prompt" & "completion" fields:
        elif "prompt" in ex and "completion" in ex:
            txt = ex["prompt"] + ex["completion"]

        # 3) If your dataset uses "instruction" & "response":
        elif "instruction" in ex and "response" in ex:
            # Some users prefer formatting like:
            #    "### Instruction:\n{instruction}\n### Response:\n{response}"
            #
            # But here we just concatenate directly.
            txt = ex["instruction"] + ex["response"]

        else:
            # No known key was found → raise immediately so you can fix the key names.
            raise KeyError(
                "None of ['text', ('prompt' & 'completion'), ('instruction' & 'response')] "
                f"were present in this example. Keys available: {list(ex.keys())}"
            )

        texts.append(txt)

    # Tokenize all of them in one go
    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=SEQ_LEN
    )

    input_ids      = encodings["input_ids"].to(DEVICE)       # (B, SEQ_LEN)
    attention_mask = encodings["attention_mask"].to(DEVICE)  # (B, SEQ_LEN)

    # For causal‐LM: labels = input_ids but mask out pads
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels
    }

# ─── DATALOADERS ────────────────────────────────────────────────────────────────
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=NUM_WORKERS,      # adjust if you have fewer/more CPU cores
    pin_memory=True
)

eval_loader = DataLoader(
    eval_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    drop_last=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# ─── OPTIMIZER ─────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ─── WARM-UP LOOP (no logging) ─────────────────────────────────────────────────
print(
    f"\nWarming up... "
    f"batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}, model={MODEL_NAME} "
    f"on {torch.cuda.get_device_name()}"
)
warmup_iter = 0
for batch in train_loader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    warmup_iter += 1
    if warmup_iter >= NUM_WARMUP_STEPS:
        break

# ─── MEASUREMENT LOOP (with W&B logging) ───────────────────────────────────────
print(f"\nMeasuring for {NUM_MEASURE_STEPS} steps...")
model.train()
torch.cuda.synchronize()
start_time = time.time()

step = 0
global_step = 0
for batch in train_loader:
    if step >= NUM_MEASURE_STEPS:
        break

    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Log train loss per step
    wandb.log({"train/loss": loss.item()}, step=global_step)

    step += 1
    global_step += 1

torch.cuda.synchronize()
end_time = time.time()

elapsed = end_time - start_time
it_per_sec    = NUM_MEASURE_STEPS / elapsed
tokens_per_sec = NUM_MEASURE_STEPS * BATCH_SIZE * SEQ_LEN / elapsed

# Log throughput metrics once
wandb.log({
    "train/it_per_sec": it_per_sec,
    "train/tokens_per_sec": tokens_per_sec
})

# ─── EVALUATION LOOP (average loss on validation split) ─────────────────────────
model.eval()
eval_losses = []
with torch.no_grad():
    for batch in eval_loader:
        outputs = model(**batch)
        eval_losses.append(outputs.loss.item())

avg_eval_loss = sum(eval_losses) / len(eval_losses)
wandb.log({"eval/loss": avg_eval_loss})

# ─── FINAL PRINT & FINISH ───────────────────────────────────────────────────────
print("\nRESULTS:")
print(f"Iterations/sec:  {it_per_sec:.2f} it/s")
print(f"Tokens/sec:      {tokens_per_sec:.2f} tokens/s")
print(f"Validation Loss: {avg_eval_loss:.4f}")

wandb.finish()