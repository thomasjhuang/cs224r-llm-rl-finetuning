#!/usr/bin/env python3
import os
from multiprocessing import cpu_count

from datasets import load_dataset
from transformers import AutoTokenizer

# ----------------------------
# Configuration
# ----------------------------
DATASET_NAME = "HuggingFaceTB/smol-smoltalk"
OUTPUT_DIR     = "smoltalk_tokenized"
MAX_LENGTH     = 1024        # same as your training max_length
BATCH_SIZE     = 256         # how many examples to process per batch
NUM_PROC       = max(1, cpu_count() - 2)  # leave 1–2 cores free for other tasks
# ----------------------------


# Load the Rust-backed (fast) tokenizer once at module import
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2-0.5B",
    trust_remote_code=True,
    use_fast=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id


def build_and_tokenize_batch(batch):
    """
    Takes a batch of examples (each example has a "messages" field, which is a list of {"role", "content"} dicts).
    Returns a dict with:
      - "input_ids": a list of token ID lists (truncated to MAX_LENGTH)
      - "labels":    a list of label ID lists, where prompt tokens are -100 and only the completion tokens are labeled
    """

    # Step 1: Extract prompt_texts and completion_texts for each example
    prompt_texts = []
    comp_texts   = []
    for msgs in batch["messages"]:
        # Find the last assistant message index
        idx = next((i for i in range(len(msgs) - 1, -1, -1) if msgs[i]["role"] == "assistant"), None)
        if idx is None:
            # No assistant message in this conversation; skip
            prompt_texts.append(None)
            comp_texts.append(None)
            continue

        prompt_msgs    = msgs[:idx]
        completion_msg = msgs[idx]

        # Build the prompt string using the same chat template logic
        prompt_str = tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True
        )
        # Build the completion string (append eos_token so the model learns <eos> at end)
        comp_str = completion_msg["content"].strip() + tokenizer.eos_token

        prompt_texts.append(prompt_str)
        comp_texts.append(comp_str)

    # Step 2: Batch-tokenize all prompt_texts (to compute lengths)
    # Replace None prompts with empty string so tokenizer won't crash
    prompt_texts_safe = [p if p is not None else "" for p in prompt_texts]
    enc_prompts = tokenizer(
        prompt_texts_safe,
        add_special_tokens=False,
        padding=False,
        truncation=False  # we want full prompt lengths to decide which to drop
    )

    prompt_lens = [len(ids) if prompt_texts[i] is not None else 0
                   for i, ids in enumerate(enc_prompts["input_ids"])]

    # Step 3: Filter out examples where prompt length >= MAX_LENGTH - 1
    valid_indices = [
        i for i, plen in enumerate(prompt_lens)
        if (prompt_texts[i] is not None) and (plen < MAX_LENGTH - 1)
    ]
    if len(valid_indices) == 0:
        return {"input_ids": [], "labels": []}

    # Build filtered lists
    filtered_prompt_texts = [prompt_texts[i] for i in valid_indices]
    filtered_comp_texts   = [comp_texts[i]   for i in valid_indices]
    filtered_plens        = [prompt_lens[i]   for i in valid_indices]

    # Step 4: Concatenate prompt + completion for each valid example, then tokenize with truncation
    full_texts = [
        filtered_prompt_texts[i] + filtered_comp_texts[i]
        for i in range(len(filtered_prompt_texts))
    ]
    enc_full = tokenizer(
        full_texts,
        add_special_tokens=False,
        padding=False,
        truncation=True,
        max_length=MAX_LENGTH
    )

    input_ids_batch = []
    labels_batch    = []
    for i, plen in enumerate(filtered_plens):
        full_ids = enc_full["input_ids"][i]
        # If truncation occurred, full_ids may be shorter than plen—we need to drop that example
        if len(full_ids) <= plen:
            # After truncation, the prompt itself took ≥ MAX_LENGTH. Skip.
            continue

        # Build labels: mask out the first plen tokens (the prompt) as -100, keep the rest
        label_ids = [-100] * plen + full_ids[plen:]
        input_ids_batch.append(full_ids)
        labels_batch.append(label_ids)

    return {"input_ids": input_ids_batch, "labels": labels_batch}


if __name__ == "__main__":
    # 1. Load the raw dataset
    ds = load_dataset(DATASET_NAME, split="train")

    # 2. Apply the batched map with Rust-based tokenization + multiple processes
    ds_tok = ds.map(
        build_and_tokenize_batch,
        batched=True,
        batch_size=BATCH_SIZE,
        remove_columns=ds.column_names,
        num_proc=NUM_PROC,
    )

    # 3. Filter out any empty examples (if build_and_tokenize_batch returned no input_ids)
    ds_tok = ds_tok.filter(lambda x: len(x["input_ids"]) > 0)

    # 4. Save the tokenized dataset to disk
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ds_tok.save_to_disk(OUTPUT_DIR)

    print(f"✅ Saved tokenized dataset to '{OUTPUT_DIR}'")