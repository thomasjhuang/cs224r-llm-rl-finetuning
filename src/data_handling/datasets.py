from typing import Dict, List, Optional, Union, Any
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from src.utils.data_utils import tokenize_with_template

class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning."""

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
            return_tensors="pt"
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
        
        try:
            prompt_str = example["prompt"]
            
            chosen_messages = example["chosen"]
            if not isinstance(chosen_messages, list) or len(chosen_messages) < 2 or not isinstance(chosen_messages[1], dict) or "content" not in chosen_messages[1]:
                raise ValueError(f"Unexpected format for 'chosen' in example: {chosen_messages}")
            chosen_completion_str = chosen_messages[1]["content"]

            rejected_messages = example["rejected"]
            if not isinstance(rejected_messages, list) or len(rejected_messages) < 2 or not isinstance(rejected_messages[1], dict) or "content" not in rejected_messages[1]:
                raise ValueError(f"Unexpected format for 'rejected' in example: {rejected_messages}")
            rejected_completion_str = rejected_messages[1]["content"]
            
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Error parsing DPO example: {example}. Original error: {e}")
        
        prompt_tokens = self.tokenizer(prompt_str, add_special_tokens=False)
        chosen_tokens = self.tokenizer(chosen_completion_str, add_special_tokens=False)
        rejected_tokens = self.tokenizer(rejected_completion_str, add_special_tokens=False)
        
        chosen_tokens['input_ids'].append(self.tokenizer.eos_token_id)
        chosen_tokens['attention_mask'].append(1)
        rejected_tokens['input_ids'].append(self.tokenizer.eos_token_id) 
        rejected_tokens['attention_mask'].append(1)
        
        chosen_sequence = {
            'input_ids': prompt_tokens['input_ids'] + chosen_tokens['input_ids'],
            'attention_mask': prompt_tokens['attention_mask'] + chosen_tokens['attention_mask']
        }
        rejected_sequence = {
            'input_ids': prompt_tokens['input_ids'] + rejected_tokens['input_ids'],
            'attention_mask': prompt_tokens['attention_mask'] + rejected_tokens['attention_mask']
        }
        
        chosen_labels = chosen_sequence['input_ids'][:]
        chosen_labels[:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
        
        rejected_labels = rejected_sequence['input_ids'][:]
        rejected_labels[:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])
        
        if len(chosen_sequence['input_ids']) > self.max_length:
            chosen_sequence['input_ids'] = chosen_sequence['input_ids'][:self.max_length]
            chosen_sequence['attention_mask'] = chosen_sequence['attention_mask'][:self.max_length]
            chosen_labels = chosen_labels[:self.max_length]
        
        if len(rejected_sequence['input_ids']) > self.max_length:
            rejected_sequence['input_ids'] = rejected_sequence['input_ids'][:self.max_length]
            rejected_sequence['attention_mask'] = rejected_sequence['attention_mask'][:self.max_length]
            rejected_labels = rejected_labels[:self.max_length]
        
        return {
            "prompt": prompt_str,
            "chosen_input_ids": torch.tensor(chosen_sequence['input_ids']),
            "chosen_attention_mask": torch.tensor(chosen_sequence['attention_mask']),
            "chosen_labels": torch.tensor(chosen_labels),
            "rejected_input_ids": torch.tensor(rejected_sequence['input_ids']),
            "rejected_attention_mask": torch.tensor(rejected_sequence['attention_mask']),
            "rejected_labels": torch.tensor(rejected_labels),
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
        
        numbers = example["nums"]
        target = example["target"]
        prompt = f"Given the numbers {', '.join(map(str, numbers))}, reach the target {target} using each number exactly once with basic arithmetic operations (+, -, *, /)."
        
        if self.tokenizer is not None:
            tokenized = tokenize_with_template(
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_length=self.max_length,
                return_tensors="pt"
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
