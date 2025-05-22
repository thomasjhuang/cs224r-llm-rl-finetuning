"""Basic tests for dataloaders without using pytest or unittest."""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer
from src.data_handling import SFTDataset, DPODataset, CountdownDataset
from src.utils.data_utils import DataCollatorForSFT, DataCollatorForDPO
from src.configs.datasets import SMOLTALK_CONFIG, ULTRAFEEDBACK_CONFIG, COUNTDOWN_CONFIG

def print_header(text):
    print("\n" + "="*80)
    print(f" {text} ".center(80, '='))
    print("="*80)

def test_sft_dataloader():
    print_header("Testing SFT Dataloader")
    
    # Initialize tokenizer
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    print("Creating SFT dataset...")
    dataset = SFTDataset(
        dataset_name=SMOLTALK_CONFIG.name,
        tokenizer=tokenizer,
        split="train[:10]",
        max_length=SMOLTALK_CONFIG.max_length,
        subset=3  
    )
    
    # Basic checks
    print(f"Dataset length: {len(dataset)}")
    print("\nFirst example keys:", list(dataset[0].keys()))
    
    # Test collator
    print("\nTesting collator...")
    collator = DataCollatorForSFT(tokenizer=tokenizer, max_length=SMOLTALK_CONFIG.max_length)
    batch = collator([dataset[i] for i in range(2)])
    print("Batch keys:", list(batch.keys()))
    print("Input IDs shape:", batch["input_ids"].shape)
    print("Labels shape:", batch["labels"].shape)
    
    print("\n SFT dataloader test passed!")

def test_dpo_dataloader():
    print_header("Testing DPO Dataloader")
    
    # Initialize tokenizer
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    print("Creating DPO dataset...")
    dataset = DPODataset(
        dataset_name=ULTRAFEEDBACK_CONFIG.name,
        tokenizer=tokenizer,
        split="train[:10]",
        max_length=ULTRAFEEDBACK_CONFIG.max_length,
        subset=3
    )
    
    # Basic checks
    print(f"Dataset length: {len(dataset)}")
    print("\nFirst example keys:", list(dataset[0].keys()))
    
    # Test collator
    print("\nTesting collator...")
    collator = DataCollatorForDPO(tokenizer=tokenizer, max_length=ULTRAFEEDBACK_CONFIG.max_length)
    batch = collator([dataset[i] for i in range(2)])
    print("Batch keys:", list(batch.keys()))
    print("Chosen input IDs shape:", batch["chosen_input_ids"].shape)
    print("Rejected input IDs shape:", batch["rejected_input_ids"].shape)
    
    print("\n DPO dataloader test passed!")

def test_countdown_dataloader():
    print_header("Testing Countdown Dataloader")
    
    # Initialize tokenizer
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    print("Creating Countdown dataset...")
    dataset = CountdownDataset(
        dataset_name=COUNTDOWN_CONFIG.name,
        tokenizer=tokenizer,
        split="train[:10]",
        max_length=COUNTDOWN_CONFIG.max_length,
        subset=3
    )
    
    # Basic checks
    print(f"Dataset length: {len(dataset)}")
    example = dataset[0]
    print("\nExample keys:", list(example.keys()))
    print("Numbers:", example["numbers"])
    print("Target:", example["target"])
    print("Prompt:", example["prompt"][:200] + "..." if len(example["prompt"]) > 200 else example["prompt"])
    
    print("\n Countdown dataloader test passed!")

if __name__ == "__main__":
    # Run all tests
    test_sft_dataloader()
    test_dpo_dataloader()
    test_countdown_dataloader()
    
    print("\n" + "="*80)
    print(" All tests completed successfully! ".center(80, '='))
    print("="*80)
