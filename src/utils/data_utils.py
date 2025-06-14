from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any
import torch
from transformers import PreTrainedTokenizerBase

@dataclass
class DataCollatorForSFT:
    """Data collator for supervised fine-tuning."""
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        processed_labels_list = []
        # Check if labels are present in the first feature and process them all if so.
        # Assumes consistent structure across features in a batch.
        first_feature_labels = features[0].get("labels") if features else None
        has_labels = first_feature_labels is not None

        if has_labels:
            for feature_dict in features:
                label_val = feature_dict.get("labels")
                if isinstance(label_val, torch.Tensor):
                    processed_labels_list.append(label_val.squeeze().tolist())
                elif isinstance(label_val, list):
                    processed_labels_list.append(label_val)
                # If label_val is None here (e.g. missing for one item), it would error later or lead to misaligned batch.
                # Current SFTDataset ensures all items have 'labels'.

        features_to_pad = []
        for feature_dict_original in features:
            item_for_padder = {}
            for key, value in feature_dict_original.items():
                if key == "labels":
                    continue
                
                if isinstance(value, torch.Tensor):
                    # Squeeze to handle both [N] and [1,N] tensors to get a 1D list
                    item_for_padder[key] = value.squeeze().tolist()
                elif isinstance(value, list):
                    item_for_padder[key] = value
                else:
                    # For other types, pass as is; tokenizer.pad might handle or raise error
                    item_for_padder[key] = value 
            features_to_pad.append(item_for_padder)

        batch = self.tokenizer.pad(
            features_to_pad,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # Ensure labels were found and processed
        if has_labels and processed_labels_list: 
            # Determine target length for labels based on padding strategy for input_ids
            target_len_for_labels = batch["input_ids"].shape[1]
            if self.padding == "max_length" and self.max_length is not None:
                target_len_for_labels = self.max_length
            
            manually_padded_labels = []
            for label_list in processed_labels_list:
                len_label = len(label_list)
                if len_label < target_len_for_labels:
                    padding_needed = target_len_for_labels - len_label
                    manually_padded_labels.append(label_list + [self.label_pad_token_id] * padding_needed)
                elif len_label > target_len_for_labels:
                    manually_padded_labels.append(label_list[:target_len_for_labels])
                else:
                    manually_padded_labels.append(label_list)
            # Ensure labels were found and processed
            if manually_padded_labels:
                batch["labels"] = torch.tensor(manually_padded_labels, dtype=torch.long)
            
        return batch

class DataCollatorForDPO:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}

        # Since sequences are now pre-padded to max_length in the dataset,
        # we just need to stack them without additional padding
        tensor_keys = [
            "chosen_input_ids", "chosen_attention_mask", "chosen_labels",
            "rejected_input_ids", "rejected_attention_mask", "rejected_labels"
        ]

        for key in tensor_keys:
            if key in features[0] and isinstance(features[0][key], torch.Tensor):
                batch[key] = torch.stack([f[key] for f in features])

        # Include prompts if they exist (as a list of strings)
        if "prompt" in features[0] and isinstance(features[0]["prompt"], str):
            batch["prompt"] = [f["prompt"] for f in features]
            
        return batch

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
    completion: Optional[str] = None,
    max_length: Optional[int] = None,
    return_tensors: Optional[str] = None
) -> Dict[str, Any]:
    messages = [
        {"role": "user", "content": prompt},
    ]
    if completion:
        messages.append({"role": "assistant", "content": completion})
    
    text = apply_chat_template(tokenizer, messages, add_generation_prompt=not bool(completion))
    
    tokenized = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=False, 
        return_attention_mask=True,
        return_tensors=return_tensors
    )
    
    # For training, we need to create labels (shifted input_ids)
    if completion:
        tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized
