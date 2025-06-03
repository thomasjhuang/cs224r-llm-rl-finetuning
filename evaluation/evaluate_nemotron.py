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
        prompt_text = extract_text_from_messages(prompt)
        response_text = extract_text_from_messages(response)
        
        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-70b-reward",
            messages=[
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": response_text}
            ]
        )
        # For reward models, the score is in the response content, not logprobs
        if hasattr(completion.choices[0], 'logprobs') and completion.choices[0].logprobs:
            if hasattr(completion.choices[0].logprobs, 'content') and completion.choices[0].logprobs.content:
                # Check if logprobs.content is a list and not empty
                if isinstance(completion.choices[0].logprobs.content, list) and completion.choices[0].logprobs.content:
                    if hasattr(completion.choices[0].logprobs.content[0], 'logprob'):
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
                
        print(f"Warning: Could not parse score from Nemotron response: {content}. Logprobs: {completion.choices[0].logprobs}")
        return 0.0 # Default score if parsing fails
        
    except Exception as e:
        print(f"Error scoring with Nemotron: {e}")
        return None

def get_vllm_response(client, prompt_text, max_tokens=512, temperature=0.7, model_name=None):
    """Get response from VLLM server"""
    try:
        if model_name is None:
            try:
                models = client.models.list()
                model_name = models.data[0].id if models.data else "dummy/model-not-found"
            except Exception as model_list_exc:
                print(f"Could not list models from VLLM, using fallback. Error: {model_list_exc}")
                model_name = "dummy/model-not-found"
        
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error getting VLLM response for model {model_name} on prompt '{prompt_text[:100]}...': {e}")
        return None

def collect_dpo_responses(args):
    """Phase 1: Collect responses from DPO model"""
    print(f"🔄 PHASE 1: Collecting responses from {args.model_type} model")
    print("=" * 60)
    
    # Test VLLM API for DPO model
    print(f"Testing VLLM API for {args.model_type} model on port {args.vllm_port}...")
    if not test_vllm_connection(args.vllm_port):
        print(f"❌ VLLM server not responding on port {args.vllm_port}")
        print(f"Make sure VLLM is running with: ./start_vllm.sh {args.model_path_to_evaluate} {args.vllm_port}")
        return False
    
    vllm_client = setup_vllm_client(args.vllm_port)
    try:
        models_list = vllm_client.models.list()
        model_name = models_list.data[0].id if models_list.data else args.model_path_to_evaluate
        print(f"✅ Connected to {args.model_type} model: {model_name}")
    except Exception as e:
        print(f"Warning: Could not query model name: {e}")
        model_name = args.model_path_to_evaluate
    
    # Load dataset
    print("Loading UltraFeedback dataset...")
    try:
        dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
        eval_samples = random.sample(list(dataset), min(args.num_samples, len(dataset)))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    print(f"Collecting responses from {len(eval_samples)} prompts...")
    
    results = []
    for i, sample in enumerate(tqdm(eval_samples)):
        prompt_text = extract_text_from_messages(sample["prompt"])
        
        dpo_response = get_vllm_response(vllm_client, prompt_text, model_name=model_name)
        if not dpo_response:
            tqdm.write(f"Failed to get {args.model_type} response for prompt {i+1}, skipping...")
            continue
        
        result = {
            "prompt": prompt_text,
            f"{args.model_type.lower()}_response": dpo_response,
            "sample_id": i
        }
        results.append(result)
        
        # Save progress every 10 samples
        if (i + 1) % 10 == 0:
            tqdm.write(f"Collected {len(results)} responses so far...")
    
    # Save DPO responses
    dpo_file = f"{args.model_type.lower()}_responses.json"
    with open(dpo_file, 'w') as f:
        json.dump({
            "model_path": args.model_path_to_evaluate,
            "model_type": args.model_type,
            "num_samples": len(results),
            "responses": results
        }, f, indent=2)
    
    print(f"\n✅ Collected {len(results)} {args.model_type} responses")
    print(f"💾 Saved to: {dpo_file}")
    print(f"\n🔄 Next step: Stop {args.model_type} model and start reference model on port {args.reference_vllm_port}")
    print(f"Then run: python {__file__} --mode collect_ref [other args]")
    return True

def collect_ref_responses(args):
    """Phase 2: Collect responses from reference model"""
    print(f"🔄 PHASE 2: Collecting responses from {args.reference_model_nickname} model")
    print("=" * 60)
    
    # Load DPO responses
    dpo_file = f"{args.model_type.lower()}_responses.json"
    if not os.path.exists(dpo_file):
        print(f"❌ DPO responses file not found: {dpo_file}")
        print("Please run --mode collect_dpo first")
        return False
        
    with open(dpo_file, 'r') as f:
        dpo_data = json.load(f)
    
    print(f"📂 Loaded {len(dpo_data['responses'])} prompts from {dpo_file}")
    
    # Test VLLM API for reference model
    print(f"Testing VLLM API for reference model on port {args.reference_vllm_port}...")
    if not test_vllm_connection(args.reference_vllm_port):
        print(f"❌ VLLM server not responding on port {args.reference_vllm_port}")
        print(f"Make sure VLLM is running with: ./start_vllm.sh {args.reference_model_name_or_path} {args.reference_vllm_port}")
        return False
    
    vllm_client = setup_vllm_client(args.reference_vllm_port)
    try:
        models_list = vllm_client.models.list()
        model_name = models_list.data[0].id if models_list.data else args.reference_model_name_or_path
        print(f"✅ Connected to reference model: {model_name}")
    except Exception as e:
        print(f"Warning: Could not query model name: {e}")
        model_name = args.reference_model_name_or_path

    print(f"Collecting reference responses for {len(dpo_data['responses'])} prompts...")
    
    # Add reference responses to existing data
    for i, sample in enumerate(tqdm(dpo_data['responses'])):
        prompt_text = sample["prompt"]
        
        ref_response = get_vllm_response(vllm_client, prompt_text, model_name=model_name)
        if not ref_response:
            tqdm.write(f"Failed to get reference response for prompt {i+1}, using fallback...")
            ref_response = "[Failed to generate response]"
            
        sample[f"{args.reference_model_nickname.lower()}_response"] = ref_response
        
        # Save progress every 10 samples
        if (i + 1) % 10 == 0:
            tqdm.write(f"Collected {i+1} reference responses so far...")
    
    # Save combined responses
    combined_file = f"{args.model_type.lower()}_vs_{args.reference_model_nickname.lower()}_responses.json"
    dpo_data["reference_model_name_or_path"] = args.reference_model_name_or_path
    dpo_data["reference_model_nickname"] = args.reference_model_nickname
    
    with open(combined_file, 'w') as f:
        json.dump(dpo_data, f, indent=2)
    
    print(f"\n✅ Collected all reference responses")
    print(f"💾 Saved to: {combined_file}")
    print(f"\n🔄 Next step: Run scoring and comparison")
    print(f"Run: python {__file__} --mode compare [other args]")
    return True

def compare_responses(args):
    """Phase 3: Score and compare all responses"""
    print(f"🔄 PHASE 3: Scoring and comparing responses")
    print("=" * 60)
    
    # Load combined responses
    combined_file = f"{args.model_type.lower()}_vs_{args.reference_model_nickname.lower()}_responses.json"
    if not os.path.exists(combined_file):
        print(f"❌ Combined responses file not found: {combined_file}")
        print("Please run --mode collect_ref first")
        return False
        
    with open(combined_file, 'r') as f:
        data = json.load(f)
    
    print(f"📂 Loaded {len(data['responses'])} response pairs from {combined_file}")
    
    # Test Nemotron API
    print("Testing Nemotron API...")
    nemotron_client = setup_nemotron_client()
    try:
        test_score = get_nemotron_score(nemotron_client, "Hello", "Hi there!")
        if test_score is not None:
            print(f"✅ Nemotron API working, test score: {test_score}")
        else:
            print("❌ Nemotron API test failed. Check API key and connectivity.")
            return False
    except Exception as e:
        print(f"❌ Nemotron API error: {e}")
        return False
    
    print(f"🎯 Scoring {len(data['responses'])} response pairs with Nemotron...")
    
    results = []
    evaluated_model_wins = 0 
    total_comparisons = 0
    
    for i, sample in enumerate(tqdm(data['responses'])):
        prompt_text = sample["prompt"]
        dpo_response = sample.get(f"{args.model_type.lower()}_response")
        ref_response = sample.get(f"{args.reference_model_nickname.lower()}_response")
        
        if not dpo_response or not ref_response:
            tqdm.write(f"Missing response for sample {i+1}, skipping...")
            continue
        
        # Score both responses
        dpo_score = get_nemotron_score(nemotron_client, prompt_text, dpo_response)
        ref_score = get_nemotron_score(nemotron_client, prompt_text, ref_response)
        
        if dpo_score is not None and ref_score is not None:
            win = 1 if dpo_score > ref_score else 0
            evaluated_model_wins += win
            total_comparisons += 1
            
            if total_comparisons > 0 and total_comparisons % 5 == 0:
                current_winrate = evaluated_model_wins / total_comparisons
                tqdm.write(f"\nProgress: {total_comparisons} scored | Current {args.model_type} win-rate: {current_winrate:.1%}")
            
            result = {
                "prompt": prompt_text,
                f"{args.model_type.lower()}_response": dpo_response,
                f"{args.reference_model_nickname.lower()}_response": ref_response,
                f"{args.model_type.lower()}_score": dpo_score,
                f"{args.reference_model_nickname.lower()}_score": ref_score,
                f"{args.model_type.lower()}_wins_vs_reference": win
            }
            results.append(result)
        else:
            tqdm.write(f"Failed to score sample {i+1}, skipping...")

        time.sleep(1) # Rate limiting for Nemotron API
    
    if total_comparisons == 0:
        print("❌ No samples were successfully compared!")
        return False
    
    # Calculate final results
    final_winrate = evaluated_model_wins / total_comparisons
    target_winrate_dpo = 0.60 
    
    status_message = ""
    if args.model_type == "DPO":
        status_message = "✅ PASS (DPO Target Met)" if final_winrate >= target_winrate_dpo else f"❌ FAIL (DPO Target Not Met - {target_winrate_dpo:.1%})"
    elif args.model_type == "SFT":
        status_message = f"📊 BASELINE (SFT win-rate vs {args.reference_model_nickname})"
    
    print("\n" + "=" * 60)
    print(f"FINAL EVALUATION RESULTS: {args.model_type} vs {args.reference_model_nickname}")
    print("=" * 60)
    print(f"Evaluated Model ({args.model_type}): {data['model_path']}")
    print(f"Reference Model ({args.reference_model_nickname}): {data['reference_model_name_or_path']}")
    print(f"Total prompts compared: {total_comparisons}")
    print(f"{args.model_type} Wins: {evaluated_model_wins}")
    print(f"{args.reference_model_nickname} Wins: {total_comparisons - evaluated_model_wins}")
    print(f"{args.model_type} Win-rate vs {args.reference_model_nickname}: {final_winrate:.1%}")
    if args.model_type == "DPO":
        print(f"Target DPO Win-rate: {target_winrate_dpo:.1%}")
    print(f"Status: {status_message}")
    print("=" * 60)
    
    # Save final results
    output_filename = f"{args.model_type.lower()}_vs_{args.reference_model_nickname.lower()}_final_results.json"
    output_data = {
        "evaluated_model_path": data['model_path'],
        "evaluated_model_type": args.model_type,
        "reference_model_name_or_path": data['reference_model_name_or_path'],
        "reference_model_nickname": args.reference_model_nickname,
        "win_rate_vs_reference": final_winrate,
        f"{args.model_type.lower()}_wins_vs_reference": evaluated_model_wins,
        f"{args.reference_model_nickname.lower()}_wins": total_comparisons - evaluated_model_wins,
        "total_comparisons": total_comparisons,
        "target_win_rate_for_dpo": target_winrate_dpo if args.model_type == "DPO" else None,
        "status": status_message,
        "samples": results
    }
    
    with open(output_filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"💾 Final results saved to: {output_filename}")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["collect_dpo", "collect_ref", "compare"], 
                       help="Evaluation mode: collect_dpo, collect_ref, or compare")
    parser.add_argument("--model_path_to_evaluate", required=True, help="Path to the primary model to evaluate (e.g., your DPO model)")
    parser.add_argument("--model_type", default="DPO", choices=["SFT", "DPO"], help="Type of the primary model being evaluated")
    parser.add_argument("--reference_model_name_or_path", default="Qwen/Qwen2.5-0.5B-Instruct", help="Name or path of the Qwen reference model")
    parser.add_argument("--reference_model_nickname", default="QwenRef", help="Short nickname for the reference model")
    parser.add_argument("--num_samples", type=int, default=25, help="Number of samples to evaluate")
    parser.add_argument("--vllm_port", type=int, default=8002, help="VLLM server port for the model to evaluate")
    parser.add_argument("--reference_vllm_port", type=int, default=8003, help="VLLM server port for the reference model")
    args = parser.parse_args()

    if args.mode == "collect_dpo":
        success = collect_dpo_responses(args)
    elif args.mode == "collect_ref":
        success = collect_ref_responses(args)
    elif args.mode == "compare":
        success = compare_responses(args)
    
    if success:
        print("✅ Phase completed successfully!")
    else:
        print("❌ Phase failed!")
        exit(1)

if __name__ == "__main__":
    main() 