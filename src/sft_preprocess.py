from datasets import load_dataset
from transformers import AutoTokenizer
import multiprocessing as mp

ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
tok.pad_token = tok.eos_token

def build_and_tokenize(example):
    msgs = example["messages"]
    idx = next((i for i in range(len(msgs)-1, -1, -1) if msgs[i]["role"] == "assistant"), None)
    if idx is None:
        return {}
    prompt, completion = msgs[:idx], msgs[idx]
    prompt_ids = tok(
        tok.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True),
        add_special_tokens=False
    )["input_ids"]
    if len(prompt_ids) >= 1023:
        return {}
    comp_tok = tok(
        completion["content"].strip() + tok.eos_token,
        add_special_tokens=False,
        truncation=True,
        max_length=1024 - len(prompt_ids)
    )["input_ids"]
    if len(comp_tok) == 0:
        return {}
    return {
        "input_ids": prompt_ids + comp_tok,
        "labels": [-100] * len(prompt_ids) + comp_tok
    }

ds_tok = ds.map(
    build_and_tokenize,
    remove_columns=ds.column_names,
    num_proc=mp.cpu_count()
)
ds_tok = ds_tok.filter(lambda x: "input_ids" in x)
ds_tok.save_to_disk("smoltalk_tokenized")