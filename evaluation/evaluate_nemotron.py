#!/usr/bin/env python3
import argparse
import os
import json
import time
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch
import random

def setup_nemotron_client():
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY", "YOUR_API_KEY_HERE")
    )
    return client

def get_nemotron_score(client, prompt, response):
    try:
        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-reward",
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
        )
        return float(completion.choices[0].logprobs.content[0].logprob)
    except Exception as e:
        print(f"Error scoring with Nemotron: {e}")
        return None

def load_models(dpo_model_path, reference_model_path):
    print(f"Loading DPO model from {dpo_model_path}...")
    dpo_tokenizer = AutoTokenizer.from_pretrained(dpo_model_path, trust_remote_code=True)
    dpo_model = AutoModelForCausalLM.from_pretrained(
        dpo_model_path, 
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"Loading reference model from {reference_model_path}...")
    ref_tokenizer = AutoTokenizer.from_pretrained(reference_model_path, trust_remote_code=True)
    ref_model = AutoModelForCausalLM.from_pretrained(
        reference_model_path,
        torch_dtype=torch.float16, 
        device_map="auto",
        trust_remote_code=True
    )
    
    return dpo_model, dpo_tokenizer, ref_model, ref_tokenizer

def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpo_model_path", default="./qwen2_dpo", help="Path to DPO trained model")
    parser.add_argument("--reference_model_path", default="Qwen/Qwen2.5-0.5B-Instruct", help="Reference model")
    parser.add_argument("--dataset_name", default="HuggingFaceH4/ultrafeedback_binarized", help="Dataset for evaluation")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--output_file", default="evaluation_results.json", help="Output file for results")
    args = parser.parse_args()
    
    client = setup_nemotron_client()
    
    dpo_model, dpo_tokenizer, ref_model, ref_tokenizer = load_models(
        args.dpo_model_path, args.reference_model_path
    )
    
    print("Loading evaluation dataset...")
    dataset = load_dataset(args.dataset_name, split="test_prefs")
    
    eval_samples = random.sample(list(dataset), min(args.num_samples, len(dataset)))
    
    results = []
    wins = 0
    total = 0
    
    for i, sample in enumerate(eval_samples):
        prompt = sample["prompt"]
        print(f"\nEvaluating sample {i+1}/{len(eval_samples)}")
        print(f"Prompt: {prompt[:100]}...")
        
        dpo_response = generate_response(dpo_model, dpo_tokenizer, prompt)
        ref_response = generate_response(ref_model, ref_tokenizer, prompt)
        
        print(f"DPO response: {dpo_response[:100]}...")
        print(f"Ref response: {ref_response[:100]}...")
        
        dpo_score = get_nemotron_score(client, prompt, dpo_response)
        ref_score = get_nemotron_score(client, prompt, ref_response)
        
        if dpo_score is not None and ref_score is not None:
            win = 1 if dpo_score > ref_score else 0
            wins += win
            total += 1
            
            result = {
                "prompt": prompt,
                "dpo_response": dpo_response,
                "ref_response": ref_response,
                "dpo_score": dpo_score,
                "ref_score": ref_score,
                "dpo_wins": win
            }
            results.append(result)
            
            current_winrate = wins / total
            print(f"Scores - DPO: {dpo_score:.3f}, Ref: {ref_score:.3f}, Win: {win}")
            print(f"Current win rate: {current_winrate:.3f} ({wins}/{total})")
        
        time.sleep(1)
    
    final_winrate = wins / total if total > 0 else 0
    print(f"\nFinal Results:")
    print(f"Win rate: {final_winrate:.3f} ({wins}/{total})")
    print(f"Target: 0.60+ for project requirements")
    
    output_data = {
        "win_rate": final_winrate,
        "wins": wins,
        "total": total,
        "results": results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main() 