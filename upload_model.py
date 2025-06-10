#!/usr/bin/env python3

from huggingface_hub import HfApi, login
import argparse
import os

def upload_model(model_path, repo_name, commit_message=None, token=None, create_repo=True):
    if not os.path.exists(model_path):
        raise ValueError(f"Model path {model_path} does not exist")
    
    api = HfApi()
    
    # Only login if a token is explicitly provided
    if token:
        login(token=token)
        print("Logged in with provided token")
    else:
        # Check if already logged in via CLI
        try:
            whoami = api.whoami()
            print(f"Already logged in as: {whoami['name']}")
        except Exception:
            raise ValueError("Not logged in. Please run: huggingface-cli login")
    
    # Create repository if it doesn't exist
    if create_repo:
        try:
            api.create_repo(
                repo_id=repo_name,
                repo_type="model",
                private=False,
                exist_ok=True
            )
            print(f"Repository {repo_name} ready")
        except Exception as e:
            print(f"Repository creation/check failed: {e}")
    
    print(f"Uploading model from {model_path} to {repo_name}...")
    
    # Default commit message based on model path
    if commit_message is None:
        if "rloo" in model_path.lower():
            commit_message = "Upload RLOO-trained Qwen2.5-0.5B model"
        elif "sft" in model_path.lower():
            commit_message = "Upload SFT-trained Qwen2.5-0.5B model"
        elif "dpo" in model_path.lower():
            commit_message = "Upload DPO-trained Qwen2.5-0.5B model"
        else:
            commit_message = "Upload trained Qwen2.5-0.5B model"
    
    api.upload_folder(
        folder_path=model_path,
        repo_id=repo_name,
        repo_type="model",
        commit_message=commit_message
    )
    
    print(f"Model uploaded successfully to: https://huggingface.co/{repo_name}")

def main():
    parser = argparse.ArgumentParser(description="Upload trained model to Hugging Face Hub")
    parser.add_argument("--model_path", "--model_dir", required=True, help="Path to the model directory")
    parser.add_argument("--repo_name", required=True, help="Hugging Face repo name (e.g., 'your-username/model-name')")
    parser.add_argument("--commit_message", help="Custom commit message (auto-detected if not provided)")
    parser.add_argument("--token", help="Hugging Face token (optional if already logged in)")
    parser.add_argument("--no-create-repo", action="store_true", help="Skip repository creation")
    args = parser.parse_args()
    
    upload_model(args.model_path, args.repo_name, args.commit_message, args.token, not args.no_create_repo)

if __name__ == "__main__":
    main() 