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
        max_prompt_length: Optional[int] = None,
        subset: Optional[int] = None,
        prompt_truncation_mode: str = "keep_end",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if max_prompt_length is None or max_prompt_length >= max_length:
            self.max_prompt_length = max_length // 2
        else:
            self.max_prompt_length = max_prompt_length
        
        if not (self.max_prompt_length < self.max_length):
            self.max_prompt_length = self.max_length // 2 
            print(f"Warning: max_prompt_length adjusted to {self.max_prompt_length} as it was not less than max_length.")

        self.prompt_truncation_mode = prompt_truncation_mode
        
        self.dataset = load_dataset(dataset_name, split=split)
        if subset is not None:
            self.dataset = self.dataset.select(range(min(len(self.dataset), subset)))
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example with careful truncation."""
        example = self.dataset[idx]
        
        try:
            prompt_str = example["prompt"]
            
            chosen_messages = example["chosen"]
            if not isinstance(chosen_messages, list) or len(chosen_messages) == 0 or \
               not isinstance(chosen_messages[0], dict) or "content" not in chosen_messages[0]:
                raise ValueError(f"Unexpected format for 'chosen' in example {idx}: {chosen_messages}. Expected list with at least one dict with 'content' key.")
            chosen_completion_str = chosen_messages[-1]["content"]

            rejected_messages = example["rejected"]
            if not isinstance(rejected_messages, list) or len(rejected_messages) == 0 or \
               not isinstance(rejected_messages[0], dict) or "content" not in rejected_messages[0]:
                raise ValueError(f"Unexpected format for 'rejected' in example {idx}: {rejected_messages}. Expected list with at least one dict with 'content' key.")
            rejected_completion_str = rejected_messages[-1]["content"]
            
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Error parsing DPO example at index {idx}: {example}. Original error: {e}")

        prompt_tok_dict = self.tokenizer(prompt_str, add_special_tokens=False, truncation=False)
        chosen_tok_dict = self.tokenizer(chosen_completion_str, add_special_tokens=False, truncation=False)
        rejected_tok_dict = self.tokenizer(rejected_completion_str, add_special_tokens=False, truncation=False)

        prompt_ids = list(prompt_tok_dict['input_ids'])
        prompt_mask = list(prompt_tok_dict['attention_mask'])
        chosen_ids = list(chosen_tok_dict['input_ids'])
        chosen_mask = list(chosen_tok_dict['attention_mask'])
        rejected_ids = list(rejected_tok_dict['input_ids'])
        rejected_mask = list(rejected_tok_dict['attention_mask'])

        chosen_ids.append(self.tokenizer.eos_token_id)
        chosen_mask.append(1)
        rejected_ids.append(self.tokenizer.eos_token_id)
        rejected_mask.append(1)

        longer_response_len = max(len(chosen_ids), len(rejected_ids))
        current_prompt_len = len(prompt_ids)

        if current_prompt_len + longer_response_len > self.max_length:
            if self.max_prompt_length < current_prompt_len:
                if self.prompt_truncation_mode == "keep_end":
                    prompt_ids = prompt_ids[-self.max_prompt_length:]
                    prompt_mask = prompt_mask[-self.max_prompt_length:]
                elif self.prompt_truncation_mode == "keep_start":
                    prompt_ids = prompt_ids[:self.max_prompt_length]
                    prompt_mask = prompt_mask[:self.max_prompt_length]
        
        prompt_len_after_trunc = len(prompt_ids)

        if prompt_len_after_trunc + len(chosen_ids) > self.max_length:
            max_chosen_comp_len = self.max_length - prompt_len_after_trunc -1
            if max_chosen_comp_len < 0 : max_chosen_comp_len = 0
            chosen_ids = chosen_ids[:max_chosen_comp_len] + [self.tokenizer.eos_token_id]
            chosen_mask = chosen_mask[:max_chosen_comp_len] + [1]

        if prompt_len_after_trunc + len(rejected_ids) > self.max_length:
            max_rejected_comp_len = self.max_length - prompt_len_after_trunc - 1
            if max_rejected_comp_len < 0: max_rejected_comp_len = 0
            rejected_ids = rejected_ids[:max_rejected_comp_len] + [self.tokenizer.eos_token_id]
            rejected_mask = rejected_mask[:max_rejected_comp_len] + [1]

        chosen_sequence_input_ids_list = prompt_ids + chosen_ids
        chosen_sequence_attention_mask_list = prompt_mask + chosen_mask
        
        rejected_sequence_input_ids_list = prompt_ids + rejected_ids
        rejected_sequence_attention_mask_list = prompt_mask + rejected_mask
        
        # Create labels as lists and mask the prompt portion
        chosen_labels_list = list(chosen_sequence_input_ids_list) # Copy list
        for i in range(min(prompt_len_after_trunc, len(chosen_labels_list))):
            chosen_labels_list[i] = -100

        rejected_labels_list = list(rejected_sequence_input_ids_list) # Copy list
        for i in range(min(prompt_len_after_trunc, len(rejected_labels_list))):
            rejected_labels_list[i] = -100

        # Final truncation of lists if sequence is too long
        if len(chosen_sequence_input_ids_list) > self.max_length:
            chosen_sequence_input_ids_list = chosen_sequence_input_ids_list[:self.max_length]
            chosen_sequence_attention_mask_list = chosen_sequence_attention_mask_list[:self.max_length]
            chosen_labels_list = chosen_labels_list[:self.max_length]

        if len(rejected_sequence_input_ids_list) > self.max_length:
            rejected_sequence_input_ids_list = rejected_sequence_input_ids_list[:self.max_length]
            rejected_sequence_attention_mask_list = rejected_sequence_attention_mask_list[:self.max_length]
            rejected_labels_list = rejected_labels_list[:self.max_length]
            
        return {
            "prompt": prompt_str, 
            "chosen_input_ids": torch.tensor(chosen_sequence_input_ids_list, dtype=torch.long),
            "chosen_attention_mask": torch.tensor(chosen_sequence_attention_mask_list, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels_list, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_sequence_input_ids_list, dtype=torch.long),
            "rejected_attention_mask": torch.tensor(rejected_sequence_attention_mask_list, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels_list, dtype=torch.long),
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
