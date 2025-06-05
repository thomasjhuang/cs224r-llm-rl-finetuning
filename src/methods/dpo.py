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
        device: Optional[torch.device] = None,
        pad_token_id: Optional[int] = None,
    ):
        self.policy = policy
        self.reference_model = reference_model
        self.beta = beta
        self.pad_token_id = pad_token_id
        
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
        """Shift labels and logits, then compute log probabilities of labels under model logits."""
        labels = labels.to(input_ids.device)

        if hasattr(self, 'use_cuda_amp') and self.use_cuda_amp and input_ids.is_cuda: # Check if use_cuda_amp exists and is True, and inputs are on CUDA
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model_instance(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                model_logits_raw = outputs.logits.to(torch.float32) # Cast to float32 for stability if mixed precision is used for forward pass
        else:
            outputs = model_instance(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            model_logits_raw = outputs.logits
        
        # Shift logits and labels for autoregressive loss calculation
        # model_logits_raw: (batch_size, seq_len, vocab_size)
        # labels: (batch_size, seq_len)
        shifted_logits = model_logits_raw[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()

        # Calculate log probabilities using log_softmax and gather, similar to original DPO implementations
        # log_softmax_logits shape: (batch_size, seq_len - 1, vocab_size)
        log_softmax_logits = F.log_softmax(shifted_logits, dim=-1)
        
        # Create a mask for valid, non-padded label positions
        # loss_mask shape: (batch_size, seq_len - 1)
        loss_mask = (shifted_labels != -100)
        
        # Prepare labels for gather: set ignored indices to 0 (won't affect sum due to loss_mask)
        # and ensure it's LongTensor for gather.
        # temp_shifted_labels shape: (batch_size, seq_len - 1)
        temp_shifted_labels = shifted_labels.clone().long() # Ensure long type
        temp_shifted_labels[~loss_mask] = 0 

        # Gather the log probabilities of the true tokens
        # true_token_logps shape: (batch_size, seq_len - 1)
        true_token_logps = torch.gather(log_softmax_logits, dim=2, index=temp_shifted_labels.unsqueeze(2)).squeeze(2)
        
        # Sum log probabilities for non-masked tokens
        # sum_logps shape: (batch_size,)
        sum_logps = (true_token_logps * loss_mask).sum(dim=-1)
        
        return sum_logps

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        chosen_input_ids = batch["chosen_input_ids"].to(self.device)
        chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
        chosen_labels = batch["chosen_labels"].to(self.device)
        rejected_input_ids = batch["rejected_input_ids"].to(self.device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)
        rejected_labels = batch["rejected_labels"].to(self.device)

        batch_size = chosen_input_ids.size(0)

        # Sequences are now pre-padded to max_length, so we can directly concatenate
        policy_input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
        policy_attention_mask = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)
        policy_labels = torch.cat([chosen_labels, rejected_labels], dim=0)
        
        all_policy_logps = self._get_batch_logps(self.policy, policy_input_ids, policy_attention_mask, policy_labels)
        policy_chosen_logps = all_policy_logps[:batch_size]
        policy_rejected_logps = all_policy_logps[batch_size:]

        with torch.no_grad():
            reference_input_ids = torch.cat([chosen_input_ids, rejected_input_ids], dim=0)
            reference_attention_mask = torch.cat([chosen_attention_mask, rejected_attention_mask], dim=0)
            reference_labels = torch.cat([chosen_labels, rejected_labels], dim=0)

            all_reference_logps = self._get_batch_logps(self.reference_model, reference_input_ids, reference_attention_mask, reference_labels)
            reference_chosen_logps = all_reference_logps[:batch_size]
            reference_rejected_logps = all_reference_logps[batch_size:]

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        loss = -F.logsigmoid(self.beta * logits).mean()

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
            "logits_std": logits.std().item() if logits.numel() > 1 and not torch.isnan(logits.std()) else 0.0,
            "policy_chosen_logp": policy_chosen_logps_mean.item(),
            "policy_rejected_logp": policy_rejected_logps_mean.item(),
            "ref_chosen_logp": reference_chosen_logps_mean.item(),
            "ref_rejected_logp": reference_rejected_logps_mean.item(),
            "chosen_rewards_mean": chosen_rewards_val.mean().item(),
            "rejected_rewards_mean": rejected_rewards_val.mean().item(),
            "reward_accuracies": (chosen_rewards_val > rejected_rewards_val).float().mean().item(),
            "reward_margins_mean": (chosen_rewards_val - rejected_rewards_val).mean().item(),
            "pi_logratios_mean": pi_logratios.mean().item(),
            "ref_logratios_mean": ref_logratios.mean().item(),
        }

        return loss, metrics
