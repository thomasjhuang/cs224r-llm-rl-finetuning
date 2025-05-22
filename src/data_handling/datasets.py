from typing import Dict, List, Optional, Union, Any
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from ..utils.data_utils import tokenize_with_template

class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizerBase,
        split: str = "train",
        max_length: int = 1024,
        subset: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.dataset = load_dataset(dataset_name, split=split)
        if subset is not None:
            self.dataset = self.dataset.select(range(min(len(self.dataset), subset)))
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        example = self.dataset[idx]
        
        if "messages" in example:
            messages = example["messages"]
            prompt = messages[0]["content"]
            completion = messages[1]["content"] if len(messages) > 1 else ""
        elif "prompt" in example and "completion" in example:
            prompt = example["prompt"]
            completion = example["completion"]
        else:
            raise ValueError(f"Unsupported dataset format: {example.keys()}")
        
        tokenized = tokenize_with_template(
            tokenizer=self.tokenizer,
            prompt=prompt,
            completion=completion,
            max_length=self.max_length,
        )
        
        return tokenized


class DPODataset(Dataset):
    """Dataset for Direct Preference Optimization."""

    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizerBase,
        split: str = "train",
        max_length: int = 1024,
        subset: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.dataset = load_dataset(dataset_name, split=split)
        if subset is not None:
            self.dataset = self.dataset.select(range(min(len(self.dataset), subset)))
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        example = self.dataset[idx]
        
        if "prompt" in example and "chosen" in example and "rejected" in example:
            prompt = example["prompt"]
            chosen = example["chosen"]
            rejected = example["rejected"]
        else:
            raise ValueError(f"Unsupported dataset format: {example.keys()}")
        
        chosen_tokenized = tokenize_with_template(
            tokenizer=self.tokenizer,
            prompt=prompt,
            completion=chosen,
            max_length=self.max_length,
        )
        
        rejected_tokenized = tokenize_with_template(
            tokenizer=self.tokenizer,
            prompt=prompt,
            completion=rejected,
            max_length=self.max_length,
        )
        
        return {
            "prompt": prompt,
            "chosen_input_ids": chosen_tokenized["input_ids"],
            "rejected_input_ids": rejected_tokenized["input_ids"],
        }


class CountdownDataset(Dataset):
    """Dataset for Countdown task with rule-based rewards."""
    
    def __init__(
        self,
        dataset_name: str = "Jiayi-Pan/Countdown-Tasks-3to4",
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        split: str = "train",
        max_length: int = 1024,
        subset: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.dataset = load_dataset(dataset_name, split=split)
        if subset is not None:
            self.dataset = self.dataset.select(range(min(len(self.dataset), subset)))
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training example."""
        example = self.dataset[idx]
        
        numbers = example["numbers"]
        target = example["target"]
        prompt = f"Given the numbers {', '.join(map(str, numbers))}, reach the target {target} using each number exactly once with basic arithmetic operations (+, -, *, /)."
        
        if self.tokenizer is not None:
            tokenized = tokenize_with_template(
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_length=self.max_length,
            )
            return {
                "prompt": prompt,
                "numbers": numbers,
                "target": target,
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
            }
        
        return {
            "prompt": prompt,
            "numbers": numbers,
            "target": target,
        }
