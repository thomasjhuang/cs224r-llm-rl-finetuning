# REINFORCE Leave-One-Out (RLOO) implementation

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import PreTrainedModel, PreTrainedTokenizer
import wandb
import logging

from src.utils.countdown_reward import calculate_countdown_reward

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
        all_texts = [] 
        all_actions = []     
        all_data = [] 

        self.model.eval() 
        with torch.no_grad():
            for i in range(batch_size):
                curr_prompt_ids_full = prompt_ids[i:i+1] 
                curr_mask_full = prompt_mask[i:i+1]
                curr_data_item = original_data[i]
                
                prompt_len = curr_prompt_ids_full.shape[1]
                
                rloo_sample_texts = []
                rloo_sample_actions = []
                rloo_sample_data_copies = []

                for _ in range(self.k_samples): # Outer RLOO k_samples loop
                    if self.ttc_internal_samples_n > 1 and self.ttc_lambda_cost > 0: # Perform internal sampling only if cost is applied
                        best_internal_reward = -float('inf')
                        best_internal_action_ids = None
                        best_internal_text = "" # Default empty string

                        for _ in range(self.ttc_internal_samples_n): # Inner TTC internal_samples_n loop
                            internal_outputs = self.model.generate(
                                curr_prompt_ids_full,
                                attention_mask=curr_mask_full,
                                max_new_tokens=self.max_length_generation,
                                temperature=self.temperature,
                                do_sample=True,
                                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                                eos_token_id=self.tokenizer.eos_token_id,
                                num_return_sequences=1,
                            )
                            internal_generated_ids = internal_outputs[:, prompt_len:] 
                            # Handle case where nothing is generated (empty tensor)
                            if internal_generated_ids.numel() == 0:
                                internal_generated_text = ""
                                current_internal_reward = calculate_countdown_reward(internal_generated_text, curr_data_item['nums'], curr_data_item['target'])[0]
                            else:
                                internal_generated_text = self.tokenizer.decode(internal_generated_ids[0], skip_special_tokens=True)
                                current_internal_reward = calculate_countdown_reward(internal_generated_text, curr_data_item['nums'], curr_data_item['target'])[0]

                            if current_internal_reward > best_internal_reward:
                                best_internal_reward = current_internal_reward
                                best_internal_action_ids = internal_generated_ids[0].to(self.device) if internal_generated_ids.numel() > 0 else torch.tensor([], dtype=torch.long, device=self.device)
                                best_internal_text = internal_generated_text
                        
                        # After internal sampling, the best one becomes the sample for RLOO
                        rloo_sample_texts.append(best_internal_text)
                        rloo_sample_actions.append(best_internal_action_ids)
                        rloo_sample_data_copies.append(curr_data_item)
                    else:
                        # Standard single generation for this RLOO k_sample
                        outputs = self.model.generate(
                            curr_prompt_ids_full,
                            attention_mask=curr_mask_full,
                            max_new_tokens=self.max_length_generation,
                            temperature=self.temperature,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            num_return_sequences=1,
                        )
                        generated_ids = outputs[:, prompt_len:]
                        if generated_ids.numel() == 0:
                            generated_text = ""
                            action_tensor = torch.tensor([], dtype=torch.long, device=self.device)
                        else:
                            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                            action_tensor = generated_ids[0].to(self.device)
                        
                        rloo_sample_texts.append(generated_text)
                        rloo_sample_actions.append(action_tensor)
                        rloo_sample_data_copies.append(curr_data_item)

                all_texts.append(rloo_sample_texts)
                all_actions.append(rloo_sample_actions)
                all_data.append(rloo_sample_data_copies)
        
        self.model.train() 
        return all_texts, all_actions, all_data

    def _get_log_probs(self, prompt_ids, prompt_mask, batch_actions):
        self.model.train()
        batch_log_probs = []

        for i in range(prompt_ids.size(0)):
            curr_prompt = prompt_ids[i]
            curr_mask = prompt_mask[i]
            prompt_len = curr_mask.sum().item()
            
            log_probs = []
            for seq_ids in batch_actions[i]:
                if len(seq_ids) == 0:
                    log_probs.append(torch.tensor(0.0, device=self.device, requires_grad=True))
                    continue

                input_ids = torch.cat([curr_prompt[:prompt_len], seq_ids], dim=0).unsqueeze(0)
                attention_mask = torch.ones_like(input_ids, device=self.device)
                
                labels = torch.full_like(input_ids, -100, device=self.device)
                labels[:, prompt_len:] = seq_ids.unsqueeze(0)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                num_tokens = seq_ids.numel()
                if num_tokens > 0:
                    sum_log_probs = -loss * num_tokens 
                else:
                    sum_log_probs = torch.tensor(0.0, device=self.device, requires_grad=True)
                
                log_probs.append(sum_log_probs)
            batch_log_probs.append(log_probs)
        return batch_log_probs

    def _compute_rewards(self, batch_texts, batch_data):
        batch_rewards = []
        batch_actual_rewards = []

        cost_penalty = 0.0
        if self.ttc_lambda_cost > 0 and self.ttc_internal_samples_n > 0:
            cost_penalty = self.ttc_lambda_cost * self.ttc_internal_samples_n 

        for i in range(len(batch_texts)):
            rewards = []
            actual_rewards = []
            for j in range(len(batch_texts[i])):
                text = batch_texts[i][j]
                data = batch_data[i][j]
                numbers = data['nums'] 
                target = data['target']
                
                actual_reward_val, _, _, _ = calculate_countdown_reward(text, numbers, target)
                actual_rewards.append(actual_reward_val)

                adjusted_reward_val = actual_reward_val - cost_penalty
                rewards.append(adjusted_reward_val)

            batch_rewards.append(rewards)
            batch_actual_rewards.append(actual_rewards)
        return batch_rewards, batch_actual_rewards

    def train_step(self, batch_prompts, batch_masks, batch_texts, batch_data):
        all_texts, all_actions, all_data = self._generate_samples(batch_prompts, batch_masks, batch_data)
        
        batch_rewards, batch_actual_rewards = self._compute_rewards(all_texts, all_data)

        all_log_probs = self._get_log_probs(batch_prompts, batch_masks, all_actions)

        policy_loss = torch.tensor(0.0, device=self.device)
        total_adjusted_reward = 0.0
        total_actual_reward = 0.0
        num_samples = 0
        
        batch_size = len(batch_rewards)
        if batch_size == 0:
            return {"loss": 0.0, "avg_reward": 0.0, "avg_actual_reward": 0.0}

        for i in range(batch_size):
            log_probs = all_log_probs[i]  
            rewards = batch_rewards[i]
            actual_rewards_for_prompt = batch_actual_rewards[i]

            if not log_probs or not rewards:
                continue

            if self.k_samples <= 1:
                if log_probs and rewards:
                    policy_loss = policy_loss - log_probs[0] * rewards[0] 
                    total_adjusted_reward += rewards[0]
                    total_actual_reward += actual_rewards_for_prompt[0]
                    num_samples += 1
            else:
                for j in range(min(len(log_probs), len(rewards))):
                    curr_reward = rewards[j]
                    curr_actual_reward = actual_rewards_for_prompt[j]
                    curr_log_prob = log_probs[j]
                    
                    baseline = 0.0
                    if self.k_samples > 1 and (len(rewards) - 1) > 0:
                        baseline = (sum(rewards) - curr_reward) / (len(rewards) - 1)
                    
                    advantage = curr_reward - baseline
                    policy_loss = policy_loss - curr_log_prob * advantage 
                    
                    total_adjusted_reward += curr_reward
                    total_actual_reward += curr_actual_reward
                    num_samples += 1
        
        avg_adjusted_reward = 0.0
        avg_actual_reward = 0.0
        avg_internal_samples = self.ttc_internal_samples_n

        if num_samples > 0:
            policy_loss = policy_loss / num_samples 
            avg_adjusted_reward = total_adjusted_reward / num_samples
            avg_actual_reward = total_actual_reward / num_samples
        
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
            "avg_reward": avg_adjusted_reward
        }
        if self.ttc_lambda_cost > 0:
            metrics["avg_actual_reward"] = avg_actual_reward
            metrics["avg_internal_samples_used"] = float(avg_internal_samples)

        return metrics

    def save_model(self, output_dir):
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
