#!/usr/bin/env python3
import argparse
import os
import json
import time
from openai import OpenAI
from datasets import load_dataset
import random
from tqdm import tqdm
import requests
import re

def setup_nemotron_client():
    """Setup Nemotron API client"""
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY", "YOUR_API_KEY_HERE")
    )
    return client

def setup_vllm_client(port=8002):
    """Setup VLLM client"""
    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="dummy-key"
    )
    return client

def test_vllm_connection(port=8002):
    """Test if VLLM server is accessible"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def extract_text_from_messages(content):
    """Extract plain text from UltraFeedback message format"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        # Extract assistant's response from message list
        for msg in content:
            if isinstance(msg, dict) and msg.get('role') == 'assistant':
                return msg.get('content', '')
        # If no assistant message found, return the last message content
        if content and isinstance(content[-1], dict):
            return content[-1].get('content', '')
    return str(content)

def get_nemotron_score(client, prompt, response):
    """Get reward score from Nemotron 70B"""
    try:
        # Ensure we have plain text strings
        prompt = extract_text_from_messages(prompt)
        response = extract_text_from_messages(response)
        
        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-reward",
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
        )
        # For reward models, the score is in the response content, not logprobs
        if hasattr(completion.choices[0], 'logprobs') and completion.choices[0].logprobs:
            # Try the logprobs approach first
            if hasattr(completion.choices[0].logprobs, 'content') and completion.choices[0].logprobs.content:
                return float(completion.choices[0].logprobs.content[0].logprob)
        
        # Fallback: parse score from message content
        content = completion.choices[0].message.content
        if content:
            try:
                # Try to extract numerical score from content
                score_match = re.search(r'[-+]?\d*\.?\d+', content)
                if score_match:
                    return float(score_match.group())
            except:
                pass
                
        # Last resort: return a default score
        print(f"Warning: Could not parse score from response: {content}")
        return 0.0
        
    except Exception as e:
        print(f"Error scoring with Nemotron: {e}")
        return None

def get_vllm_response(client, prompt, max_tokens=512, temperature=0.7, model_name=None):
    """Get response from VLLM server"""
    try:
        # Try to get available models if model_name not provided
        if model_name is None:
            try:
                models = client.models.list()
                model_name = models.data[0].id if models.data else "dummy"
            except:
                model_name = "dummy"
        
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error getting VLLM response: {e}")
        return None

def main():
    print("DPO vs UltraFeedback Evaluation using Nemotron 70B")
    print("=" * 60)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpo_model", help="Path to DPO model (for reference, VLLM should already be serving it)")
    parser.add_argument("--num_samples", type=int, default=25, help="Number of samples to evaluate")
    parser.add_argument("--vllm_port", type=int, default=8002, help="VLLM server port")
    parser.add_argument("--output_file", default="dpo_vs_ultrafeedback_nemotron_results.json", help="Output file")
    args = parser.parse_args()
    
    # Test Nemotron API
    print("Testing Nemotron API...")
    nemotron_client = setup_nemotron_client()
    try:
        test_score = get_nemotron_score(nemotron_client, "Hello", "Hi there!")
        if test_score is not None:
            print(f"✅ Nemotron API working, test score: {test_score}")
        else:
            print("❌ Nemotron API test failed")
            return
    except Exception as e:
        print(f"❌ Nemotron API error: {e}")
        return
    
    # Test VLLM API
    print("Testing local VLLM API...")
    if not test_vllm_connection(args.vllm_port):
        print(f"❌ VLLM server not responding on port {args.vllm_port}")
        print("Make sure VLLM is running with: ./start_vllm.sh <model_path>")
        return
    
    vllm_client = setup_vllm_client(args.vllm_port)
    try:
        test_response = get_vllm_response(vllm_client, "Hello", max_tokens=50)
        if test_response:
            print(f"✅ VLLM API working, test response: {test_response[:50]}...")
        else:
            print("❌ VLLM API test failed")
            return
    except Exception as e:
        print(f"❌ VLLM API error: {e}")
        return
    
    # Load dataset
    print("Loading UltraFeedback dataset...")
    try:
        dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
        eval_samples = random.sample(list(dataset), min(args.num_samples, len(dataset)))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    print(f"Evaluating {len(eval_samples)} samples...")
    
    # Evaluation loop
    results = []
    dpo_wins = 0
    total_evaluated = 0
    
    for i, sample in enumerate(tqdm(eval_samples)):
        prompt = sample["prompt"]
        ultrafeedback_chosen = sample["chosen"]
        
        print(f"\n--- Sample {i+1}/{len(eval_samples)} ---")
        print(f"Prompt: {prompt[:100]}...")
        
        # Get DPO response
        dpo_response = get_vllm_response(vllm_client, prompt)
        if not dpo_response:
            print("Failed to get DPO response, skipping...")
            continue
        
        print(f"DPO Response: {dpo_response[:100]}...")
        print(f"UltraFeedback Chosen: {ultrafeedback_chosen[:100]}...")
        
        # Get Nemotron scores
        print("Getting Nemotron scores...")
        dpo_score = get_nemotron_score(nemotron_client, prompt, dpo_response)
        uf_score = get_nemotron_score(nemotron_client, prompt, ultrafeedback_chosen)
        
        if dpo_score is not None and uf_score is not None:
            win = 1 if dpo_score > uf_score else 0
            dpo_wins += win
            total_evaluated += 1
            
            winner = "DPO" if win else "UltraFeedback"
            print(f"DPO Score: {dpo_score:.3f}")
            print(f"UltraFeedback Score: {uf_score:.3f}")
            print(f"Winner: {winner}")
            
            if total_evaluated % 5 == 0:
                current_winrate = dpo_wins / total_evaluated
                print(f"\nProgress: {total_evaluated}/{len(eval_samples)} | Current win-rate: {current_winrate:.1%}")
            
            result = {
                "prompt": prompt,
                "dpo_response": dpo_response,
                "ultrafeedback_chosen": ultrafeedback_chosen,
                "dpo_score": dpo_score,
                "ultrafeedback_score": uf_score,
                "dpo_wins": win
            }
            results.append(result)
        
        # Rate limiting
        time.sleep(1)
    
    if total_evaluated == 0:
        print("No samples were successfully evaluated!")
        return
    
    # Final results
    final_winrate = dpo_wins / total_evaluated
    target_winrate = 0.60
    status = "✅ PASS" if final_winrate >= target_winrate else "❌ FAIL"
    
    print("\n" + "=" * 50)
    print("ULTRAFEEDBACK EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total samples evaluated: {total_evaluated}")
    print(f"DPO Wins: {dpo_wins}")
    print(f"UltraFeedback Wins: {total_evaluated - dpo_wins}")
    print(f"DPO Win-rate: {final_winrate:.1%}")
    print(f"Target: {target_winrate:.1%}")
    print(f"Status: {status}")
    print("=" * 50)
    
    # Save results
    output_data = {
        "win_rate": final_winrate,
        "dpo_wins": dpo_wins,
        "total_evaluated": total_evaluated,
        "target_win_rate": target_winrate,
        "status": "pass" if final_winrate >= target_winrate else "fail",
        "samples": results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nDetailed results saved to: {args.output_file}")

if __name__ == "__main__":
    main() 