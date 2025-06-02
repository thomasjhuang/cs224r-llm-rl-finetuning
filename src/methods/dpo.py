# Direct Preference Optimization (DPO) implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from transformers import PreTrainedModel

class DPOMethod:
    def __init__(
        self,
        model: PreTrainedModel,
        ref_model: PreTrainedModel,
        beta: float = 0.1,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.ref_model = ref_model
        self.beta = beta
        
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        self.use_cuda_amp = self.device.type == "cuda"
        
        for param in self.ref_model.parameters():
            param.requires_grad = False
        self.ref_model.eval()

    def get_log_probs(self, model: PreTrainedModel, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.use_cuda_amp:
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
        log_probs = F.log_softmax(logits, dim=-1)
        labels = input_ids[:, 1:].clone()
        log_probs = log_probs[:, :-1, :]
        
        per_token_log_probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)
        
        mask = attention_mask[:, 1:].clone()
        per_token_log_probs = per_token_log_probs * mask
        
        sequence_log_probs = per_token_log_probs.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        return sequence_log_probs

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        chosen_input_ids = batch["chosen_input_ids"].to(self.device)
        chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
        rejected_input_ids = batch["rejected_input_ids"].to(self.device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)

        policy_chosen_log_probs = self.get_log_probs(self.model, chosen_input_ids, chosen_attention_mask)
        policy_rejected_log_probs = self.get_log_probs(self.model, rejected_input_ids, rejected_attention_mask)

        with torch.no_grad():
            ref_chosen_log_probs = self.get_log_probs(self.ref_model, chosen_input_ids, chosen_attention_mask)
            ref_rejected_log_probs = self.get_log_probs(self.ref_model, rejected_input_ids, rejected_attention_mask)

        policy_chosen_ratio = policy_chosen_log_probs - ref_chosen_log_probs
        policy_rejected_ratio = policy_rejected_log_probs - ref_rejected_log_probs

        logits = self.beta * (policy_chosen_ratio - policy_rejected_ratio)
        loss = -F.logsigmoid(logits).mean()

        return loss
