# REINFORCE Leave-One-Out (RLOO) implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import PreTrainedModel, PreTrainedTokenizer
import wandb
import logging

from src.utils.countdown_reward import calculate_countdown_reward
from src.utils.countdown import compute_score

logger = logging.getLogger(__name__)

class RLOOTrainer:
    def __init__(self, model, tokenizer, optimizer, lr_scheduler=None, k_samples=4, max_length_generation=128, temperature=0.7, gamma=1.0, device=None, gradient_accumulation_steps=1, ttc_internal_samples_n=1, ttc_lambda_cost=0.0):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.k_samples = k_samples
        self.max_length_generation = max_length_generation
        self.temperature = temperature
        self.gamma = gamma
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.grad_accum_count = 0
        self.ttc_internal_samples_n = ttc_internal_samples_n
        self.ttc_lambda_cost = ttc_lambda_cost

    def _generate_samples(self, prompt_ids, prompt_mask, original_data):
        batch_size = prompt_ids.size(0)
        self.model.eval()

        # Determine number of generations per prompt
        is_ttc = self.ttc_internal_samples_n > 1 and self.ttc_lambda_cost > 0
        num_gens_per_prompt = self.k_samples * (self.ttc_internal_samples_n if is_ttc else 1)

        # Repeat prompts and masks for batch generation
        expanded_prompt_ids = prompt_ids.repeat_interleave(num_gens_per_prompt, dim=0)
        expanded_prompt_mask = prompt_mask.repeat_interleave(num_gens_per_prompt, dim=0)
        
        prompt_len = prompt_ids.shape[1]

        with torch.no_grad():
            # Use the underlying model's generate function if using DataParallel
            model_to_generate = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            generated_outputs = model_to_generate.generate(
                input_ids=expanded_prompt_ids,
                attention_mask=expanded_prompt_mask,
                                max_new_tokens=self.max_length_generation,
                                temperature=self.temperature,
                                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = generated_outputs[:, prompt_len:]
        generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        all_texts, all_actions, all_data = [], [], []
        gen_idx = 0
        for i in range(batch_size):
            curr_data_item = original_data[i]
            rloo_sample_texts, rloo_sample_actions, rloo_sample_data_copies = [], [], []

            for _ in range(self.k_samples):
                if is_ttc:
                    internal_texts = generated_texts[gen_idx : gen_idx + self.ttc_internal_samples_n]
                    internal_ids = generated_ids[gen_idx : gen_idx + self.ttc_internal_samples_n]
                    gen_idx += self.ttc_internal_samples_n
                    
                    rewards = [calculate_countdown_reward(t, curr_data_item['nums'], curr_data_item['target'])[0] for t in internal_texts]
                    best_idx = rewards.index(max(rewards))
                    
                    rloo_sample_texts.append(internal_texts[best_idx])
                    rloo_sample_actions.append(internal_ids[best_idx].to(self.device))
                else:
                    rloo_sample_texts.append(generated_texts[gen_idx])
                    rloo_sample_actions.append(generated_ids[gen_idx].to(self.device))
                    gen_idx += 1
                
                rloo_sample_data_copies.append(curr_data_item)

            all_texts.append(rloo_sample_texts)
            all_actions.append(rloo_sample_actions)
            all_data.append(rloo_sample_data_copies)
        
        self.model.train() 
        return all_texts, all_actions, all_data

    def _get_log_probs(self, prompt_ids, prompt_mask, batch_actions):
        self.model.train()
        
        all_input_ids, all_labels = [], []
        prompt_lengths = [m.sum().item() for m in prompt_mask]

        for i in range(prompt_ids.size(0)):
            for seq_ids in batch_actions[i]:
                if len(seq_ids) == 0: continue
                prompt_len = prompt_lengths[i]
                curr_prompt = prompt_ids[i, :prompt_len]
                
                input_ids = torch.cat([curr_prompt, seq_ids], dim=0)
                labels = torch.full_like(input_ids, -100)
                labels[prompt_len:] = seq_ids
                
                all_input_ids.append(input_ids)
                all_labels.append(labels)

        if not all_input_ids:
            return [[torch.tensor(0.0, device=self.device, requires_grad=True)] * self.k_samples for _ in range(prompt_ids.size(0))]

        padded_input_ids = nn.utils.rnn.pad_sequence(all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)
        padded_labels = nn.utils.rnn.pad_sequence(all_labels, batch_first=True, padding_value=-100).to(self.device)
        attention_mask = (padded_input_ids != self.tokenizer.pad_token_id).long()
        
        # Handle potential DataParallel issues with uneven batch sizes
        model_to_use = self.model
        is_data_parallel = isinstance(self.model, nn.DataParallel)
        if is_data_parallel:
            num_gpus = torch.cuda.device_count()
            batch_size = padded_input_ids.size(0)
            if batch_size > 0 and (batch_size % num_gpus != 0 or batch_size < num_gpus):
                logger.warning(
                    f"Batch size {batch_size} is not suitable for {num_gpus} GPUs. "
                    f"Running this batch on a single device to avoid NCCL errors."
                )
                model_to_use = self.model.module

        outputs = model_to_use(input_ids=padded_input_ids, attention_mask=attention_mask, labels=padded_labels)
        
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = padded_labels[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_logits.size(0), -1)

        loss_mask = (shift_labels != -100)
        per_sequence_loss = (loss * loss_mask).sum(dim=1)
        sum_log_probs_flat = -per_sequence_loss

        batch_log_probs, current_idx = [], 0
        for i in range(prompt_ids.size(0)):
            log_probs = []
            for seq_ids in batch_actions[i]:
                if len(seq_ids) == 0:
                    log_probs.append(torch.tensor(0.0, device=self.device, requires_grad=True))
                else:
                    log_probs.append(sum_log_probs_flat[current_idx])
                    current_idx += 1
            batch_log_probs.append(log_probs)
            
        return batch_log_probs

    def _compute_rewards(self, batch_texts, batch_data):
        batch_rewards = []
        batch_actual_rewards = []
        batch_sparse_rewards = []

        cost_penalty = 0.0
        if self.ttc_lambda_cost > 0 and self.ttc_internal_samples_n > 0:
            cost_penalty = self.ttc_lambda_cost * self.ttc_internal_samples_n 

        for i in range(len(batch_texts)):
            rewards = []
            actual_rewards = []
            sparse_rewards = []
            for j in range(len(batch_texts[i])):
                text = batch_texts[i][j]
                data = batch_data[i][j]
                numbers = data['nums'] 
                target = data['target']
                
                ground_truth = {'numbers': numbers, 'target': target}
                
                # Use the official sparse reward for training to enforce format
                sparse_reward_val = compute_score(solution_str=text, ground_truth=ground_truth)
                
                # We will use this for both training and monitoring
                actual_reward_val = sparse_reward_val
                
                actual_rewards.append(actual_reward_val)
                sparse_rewards.append(sparse_reward_val)

                adjusted_reward_val = actual_reward_val - cost_penalty
                rewards.append(adjusted_reward_val)

            batch_rewards.append(rewards)
            batch_actual_rewards.append(actual_rewards)
            batch_sparse_rewards.append(sparse_rewards)
        return batch_rewards, batch_actual_rewards, batch_sparse_rewards

    def train_step(self, batch_prompts, batch_masks, batch_texts, batch_data):
        all_texts, all_actions, all_data = self._generate_samples(batch_prompts, batch_masks, batch_data)
        
        batch_rewards, batch_actual_rewards, batch_sparse_rewards = self._compute_rewards(all_texts, all_data)

        all_log_probs = self._get_log_probs(batch_prompts, batch_masks, all_actions)

        policy_loss = torch.tensor(0.0, device=self.device)
        total_adjusted_reward = 0.0
        total_actual_reward = 0.0
        total_sparse_reward = 0.0
        num_samples = 0
        
        batch_size = len(batch_rewards)
        if batch_size == 0:
            return {"loss": 0.0, "avg_reward": 0.0, "avg_actual_reward": 0.0, "avg_sparse_reward": 0.0}

        for i in range(batch_size):
            log_probs = all_log_probs[i]  
            rewards = batch_rewards[i]
            actual_rewards_for_prompt = batch_actual_rewards[i]
            sparse_rewards_for_prompt = batch_sparse_rewards[i]

            if not log_probs or not rewards:
                continue

            if self.k_samples <= 1:
                if log_probs and rewards:
                    policy_loss = policy_loss - log_probs[0] * rewards[0] 
                    total_adjusted_reward += rewards[0]
                    total_actual_reward += actual_rewards_for_prompt[0]
                    total_sparse_reward += sparse_rewards_for_prompt[0]
                    num_samples += 1
            else:
                for j in range(min(len(log_probs), len(rewards))):
                    curr_reward = rewards[j]
                    curr_actual_reward = actual_rewards_for_prompt[j]
                    curr_sparse_reward = sparse_rewards_for_prompt[j]
                    curr_log_prob = log_probs[j]
                    
                    baseline = 0.0
                    if self.k_samples > 1 and (len(rewards) - 1) > 0:
                        baseline = (sum(rewards) - curr_reward) / (len(rewards) - 1)
                    
                    advantage = curr_reward - baseline
                    policy_loss = policy_loss - curr_log_prob * advantage 
                    
                    total_adjusted_reward += curr_reward
                    total_actual_reward += curr_actual_reward
                    total_sparse_reward += curr_sparse_reward
                    num_samples += 1
        
        avg_adjusted_reward = 0.0
        avg_actual_reward = 0.0
        avg_sparse_reward = 0.0
        avg_internal_samples = self.ttc_internal_samples_n

        if num_samples > 0:
            policy_loss = policy_loss / num_samples 
            avg_adjusted_reward = total_adjusted_reward / num_samples
            avg_actual_reward = total_actual_reward / num_samples
            avg_sparse_reward = total_sparse_reward / num_samples
        
        if torch.is_tensor(policy_loss) and policy_loss.requires_grad:
            loss_scaled = policy_loss / self.gradient_accumulation_steps
            loss_scaled.backward()
            
            self.grad_accum_count += 1
            if self.grad_accum_count % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.grad_accum_count = 0

        metrics = {
            "loss": policy_loss.item() if torch.is_tensor(policy_loss) else float(policy_loss),
            "avg_reward": avg_adjusted_reward,
            "avg_sparse_reward": avg_sparse_reward
        }
        if self.ttc_lambda_cost > 0:
            metrics["avg_actual_reward"] = avg_actual_reward
            metrics["avg_internal_samples_used"] = float(avg_internal_samples)

        return metrics

    def save_model(self, output_dir):
        model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
