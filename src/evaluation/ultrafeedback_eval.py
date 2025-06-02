#!/usr/bin/env python
import argparse
import json
import time
from typing import List, Dict, Any
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltraFeedbackEvaluator:
    def __init__(self, api_key: str, base_url: str = "https://integrate.api.nvidia.com/v1"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = "nvidia/llama-3.1-nemotron-70b-reward"

    def get_reward_score(self, prompt: str, response: str, max_retries: int = 3) -> float:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages
                )
                
                if hasattr(completion, 'choices') and completion.choices:
                    score = float(completion.choices[0].message.content.strip())
                    return score
                else:
                    logger.warning(f"Unexpected response format: {completion}")
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"All attempts failed for prompt: {prompt[:100]}...")
                    return 0.0
        
        return 0.0

def generate_response(model, tokenizer, prompt: str, max_length: int = 1024, temperature: float = 0.7) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--reference_model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Reference model for comparison")
    parser.add_argument("--api_key", required=True, help="NVIDIA API key for Nemotron")
    parser.add_argument("--dataset_name", default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--split", default="test_prefs")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to evaluate")
    parser.add_argument("--output_file", default="ultrafeedback_results.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device(args.device)
    evaluator = UltraFeedbackEvaluator(api_key=args.api_key)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    trained_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    ).to(device).eval()
    
    ref_tokenizer = AutoTokenizer.from_pretrained(args.reference_model, trust_remote_code=True)
    ref_tokenizer.pad_token = ref_tokenizer.eos_token
    
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.reference_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    ).to(device).eval()
    
    try:
        dataset = load_dataset(args.dataset_name, split=args.split)
    except:
        dataset = load_dataset(args.dataset_name, split="test")
    
    if len(dataset) > args.max_samples:
        dataset = dataset.select(range(args.max_samples))
    
    results = []
    wins = 0
    total = 0
    
    logger.info(f"Evaluating {len(dataset)} samples...")
    
    for i, example in enumerate(dataset):
        prompt = example["prompt"]
        
        logger.info(f"Processing sample {i+1}/{len(dataset)}")
        
        trained_response = generate_response(trained_model, tokenizer, prompt)
        ref_response = generate_response(reference_model, ref_tokenizer, prompt)
        
        trained_score = evaluator.get_reward_score(prompt, trained_response)
        ref_score = evaluator.get_reward_score(prompt, ref_response)
        
        win = 1 if trained_score > ref_score else 0
        wins += win
        total += 1
        
        result = {
            "sample_id": i,
            "prompt": prompt,
            "trained_response": trained_response,
            "reference_response": ref_response,
            "trained_score": trained_score,
            "reference_score": ref_score,
            "win": win,
            "current_winrate": wins / total
        }
        
        results.append(result)
        
        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i+1}/{len(dataset)}, Current win rate: {wins/total:.3f}")
    
    final_winrate = wins / total
    
    summary = {
        "total_samples": total,
        "wins": wins,
        "winrate": final_winrate,
        "model_path": args.model_path,
        "reference_model": args.reference_model,
        "results": results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Final Results:")
    logger.info(f"Total samples: {total}")
    logger.info(f"Wins: {wins}")
    logger.info(f"Win rate: {final_winrate:.3f}")
    logger.info(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main() 