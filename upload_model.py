#!/usr/bin/env python3

from huggingface_hub import HfApi, login
import argparse

def upload_model(model_path, repo_name, token=None):
    if token:
        login(token=token)
    else:
        print("Please login to Hugging Face (run: huggingface-cli login)")
    
    api = HfApi()
    
    print(f"Uploading model from {model_path} to {repo_name}...")
    
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_name,
        repo_type="model",
        commit_message="Upload DPO-trained Qwen2.5-0.5B model"
    )
    
    print(f"Model uploaded successfully to: https://huggingface.co/{repo_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./qwen2_dpo")
    parser.add_argument("--repo_name", required=True, help="e.g., 'your-username/qwen2-dpo-ultrafeedback'")
    parser.add_argument("--token", help="Hugging Face token (optional if already logged in)")
    args = parser.parse_args()
    
    upload_model(args.model_path, args.repo_name, args.token)

if __name__ == "__main__":
    main() 