from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from datasets import Dataset

@dataclass
class DataCollatorForSFT:
    """Data collator for supervised fine-tuning."""
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 1024
    
    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate function for SFT data."""
        input_ids = [instance["input_ids"][:self.max_length] for instance in instances]
        labels = [instance["labels"][:self.max_length] for instance in instances]
        
        # Pad sequences
        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "labels": labels},
            padding="longest",
            return_tensors="pt",
        )
        
        # Create attention mask
        batch["attention_mask"] = batch["input_ids"].ne(self.tokenizer.pad_token_id).long()
        
        return batch

@dataclass
class DataCollatorForDPO:
    """Data collator for DPO training."""
    tokenizer: PreTrainedTokenizerBase
    max_length: int = 1024
    
    def __call__(self, instances: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate function for DPO data."""
        # Extract chosen and rejected inputs
        chosen_inputs = [{"input_ids": inst["chosen_input_ids"], "attention_mask": [1] * len(inst["chosen_input_ids"])} for inst in instances]
        rejected_inputs = [{"input_ids": inst["rejected_input_ids"], "attention_mask": [1] * len(inst["rejected_input_ids"])} for inst in instances]
        
        # Pad and truncate
        def prepare_inputs(inputs):
            return self.tokenizer.pad(
                inputs,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
                pad_to_multiple_of=8,
            )
        
        chosen_batch = prepare_inputs(chosen_inputs)
        rejected_batch = prepare_inputs(rejected_inputs)
        
        return {
            "chosen_input_ids": chosen_batch["input_ids"],
            "chosen_attention_mask": chosen_batch["attention_mask"],
            "rejected_input_ids": rejected_batch["input_ids"],
            "rejected_attention_mask": rejected_batch["attention_mask"],
            "return_loss": True,
        }

def apply_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False
) -> str:
    """Apply chat template to format messages."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )

def tokenize_with_template(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    completion: str = "",
    max_length: int = 1024,
    truncation: bool = True,
    padding: bool = False
) -> Dict[str, torch.Tensor]:
    """Tokenize input with chat template."""
    messages = [
        {"role": "user", "content": prompt},
    ]
    if completion:
        messages.append({"role": "assistant", "content": completion})
    
    text = apply_chat_template(tokenizer, messages, add_generation_prompt=not bool(completion))
    
    tokenized = tokenizer(
        text,
        max_length=max_length,
        truncation=truncation,
        padding=padding,
        return_tensors="pt" if padding else None,
    )
    
    # For training, we need to create labels (shifted input_ids)
    if completion:
        tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized
