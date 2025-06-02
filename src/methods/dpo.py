# Direct Preference Optimization (DPO) implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from transformers import PreTrainedModel

class DPOMethod:
    def __init__(
        self,
        policy: PreTrainedModel,
        reference_model: PreTrainedModel,
        beta: float = 0.1,
        device: Optional[torch.device] = None
    ):
        self.policy = policy
        self.reference_model = reference_model
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
        
        for param in self.reference_model.parameters():
            param.requires_grad = False
        self.reference_model.eval()

    def _get_batch_logps(self, model_instance: PreTrainedModel, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.use_cuda_amp:
            with torch.amp.autocast('cuda'):
                outputs = model_instance(input_ids=input_ids, attention_mask=attention_mask)
                model_logits = outputs.logits
        else:
            outputs = model_instance(input_ids=input_ids, attention_mask=attention_mask)
            model_logits = outputs.logits
            
        assert model_logits.shape[:-1] == labels.shape
        
        labels = labels[:, 1:].clone()
        model_logits = model_logits[:, :-1, :]
        loss_mask = (labels != -100)
        
        labels[labels == -100] = 0
        
        per_token_log_probs = torch.gather(model_logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        per_token_log_probs = per_token_log_probs * loss_mask
        sequence_log_probs = per_token_log_probs.sum(dim=1)
        
        return sequence_log_probs

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        chosen_input_ids = batch["chosen_input_ids"].to(self.device)
        chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
        chosen_labels = batch["chosen_labels"].to(self.device)
        rejected_input_ids = batch["rejected_input_ids"].to(self.device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)
        rejected_labels = batch["rejected_labels"].to(self.device)

        policy_chosen_logps = self._get_batch_logps(self.policy, chosen_input_ids, chosen_attention_mask, chosen_labels)
        policy_rejected_logps = self._get_batch_logps(self.policy, rejected_input_ids, rejected_attention_mask, rejected_labels)

        with torch.no_grad():
            reference_chosen_logps = self._get_batch_logps(self.reference_model, chosen_input_ids, chosen_attention_mask, chosen_labels)
            reference_rejected_logps = self._get_batch_logps(self.reference_model, rejected_input_ids, rejected_attention_mask, rejected_labels)

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        loss = -F.logsigmoid(self.beta * logits).mean()

        # Add debugging metrics
        with torch.no_grad():
            policy_chosen_logps_minus_ref = policy_chosen_logps - reference_chosen_logps
            policy_rejected_logps_minus_ref = policy_rejected_logps - reference_rejected_logps
            
            policy_chosen_logps_mean = policy_chosen_logps.mean()
            policy_rejected_logps_mean = policy_rejected_logps.mean()
            reference_chosen_logps_mean = reference_chosen_logps.mean()
            reference_rejected_logps_mean = reference_rejected_logps.mean()

            chosen_rewards_val = self.beta * policy_chosen_logps_minus_ref
            rejected_rewards_val = self.beta * policy_rejected_logps_minus_ref
            
        metrics = {
            "logits_std": logits.std().item(),
            "policy_chosen_logp": policy_chosen_logps_mean.item(),
            "policy_rejected_logp": policy_rejected_logps_mean.item(),
            "ref_chosen_logp": reference_chosen_logps_mean.item(),
            "ref_rejected_logp": reference_rejected_logps_mean.item(),
            "chosen_rewards_mean": chosen_rewards_val.mean().item(),
            "rejected_rewards_mean": rejected_rewards_val.mean().item(),
            "reward_accuracies": (chosen_rewards_val > rejected_rewards_val).float().mean().item(),
            "reward_margins_mean": (chosen_rewards_val - rejected_rewards_val).mean().item(),
        }

        return loss, metrics
