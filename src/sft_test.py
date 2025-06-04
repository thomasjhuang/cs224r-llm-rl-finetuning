import torch
import time
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────
MODEL_NAME        = "Qwen/Qwen2.5-0.5B"
SEQ_LEN           = 1024
BATCH_SIZE        = 64
NUM_WARMUP_STEPS  = 10
NUM_MEASURE_STEPS = 50
DEVICE            = "cuda"
LEARNING_RATE     = 1e-4
PROJECT_NAME      = "qwen2-smoltalk-sft-test"

# ─────────────────────────────────────────────────────────────────────
# INITIALIZE WANDB
# ─────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────
# LOAD TOKENIZER + MODEL
# ─────────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Ensure a pad_token exists
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to(DEVICE)
model.train()

# ─────────────────────────────────────────────────────────────────────
# LOAD AND SPLIT SMOLTALK DATASET
# ─────────────────────────────────────────────────────────────────────
raw_ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")

# Split 90% train / 10% validation
split_ds = raw_ds.train_test_split(test_size=0.10, seed=42)
train_ds = split_ds["train"]
eval_ds  = split_ds["test"]

# ─────────────────────────────────────────────────────────────────────
# COLLATE FUNCTION
# ─────────────────────────────────────────────────────────────────────
def collate_fn(batch):
    texts = [ex["text"] for ex in batch]
    encodings = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=SEQ_LEN
    )

    input_ids      = encodings["input_ids"].to(DEVICE)       # (B, SEQ_LEN)
    attention_mask = encodings["attention_mask"].to(DEVICE)  # (B, SEQ_LEN)

    # Labels: same as input_ids, but mask out pads
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels
    }

# ─────────────────────────────────────────────────────────────────────
# DATALOADERS
# ─────────────────────────────────────────────────────────────────────
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=4,
    pin_memory=True
)

eval_loader = DataLoader(
    eval_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    drop_last=False,
    num_workers=2,
    pin_memory=True
)

# ─────────────────────────────────────────────────────────────────────
# OPTIMIZER
# ─────────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ─────────────────────────────────────────────────────────────────────
# WARM‐UP LOOP (no logging)
# ─────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────
# MEASUREMENT LOOP (with W&B logging)
# ─────────────────────────────────────────────────────────────────────
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
it_per_sec = NUM_MEASURE_STEPS / elapsed
tokens_per_sec = NUM_MEASURE_STEPS * BATCH_SIZE * SEQ_LEN / elapsed

# Log throughput metrics once
wandb.log({
    "train/it_per_sec": it_per_sec,
    "train/tokens_per_sec": tokens_per_sec
})

# ─────────────────────────────────────────────────────────────────────
# EVALUATION LOOP (compute average loss on validation split)
# ─────────────────────────────────────────────────────────────────────
model.eval()
eval_losses = []
with torch.no_grad():
    for batch in eval_loader:
        outputs = model(**batch)
        eval_losses.append(outputs.loss.item())

avg_eval_loss = sum(eval_losses) / len(eval_losses)
wandb.log({"eval/loss": avg_eval_loss})

# ─────────────────────────────────────────────────────────────────────
# FINAL PRINT & FINISH
# ─────────────────────────────────────────────────────────────────────
print("\nRESULTS:")
print(f"Iterations/sec: {it_per_sec:.2f} it/s")
print(f"Tokens/sec:     {tokens_per_sec:.2f} tokens/s")
print(f"Validation Loss: {avg_eval_loss:.4f}")

wandb.finish()