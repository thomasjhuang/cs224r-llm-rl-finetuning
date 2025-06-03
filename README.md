# CS224R RL Fine-Tuning Project

This project focuses on implementing and exploring Reinforcement Learning algorithms for fine-tuning Large Language Models (LLMs), specifically Qwen 2.5 0.5B Base.

## Project Goal

The primary goal of the initial implementation is to:
1.  Set up data loading pipelines for Instruction Following (UltraFeedback, SmolTalk) and Math Reasoning (Countdown, WarmStart, TinyZero prompts) tasks.
2.  Implement Supervised Fine-Tuning (SFT).
3.  Implement Direct Preference Optimization (DPO).
4.  Implement REINFORCE Leave One-Out (RLOO) with Bradley-Terry reward modeling for preference data and a rule-based reward for math tasks.
5.  Establish an evaluation setup for both tasks using the specified metrics and models.

## Quick Start: SFT Model → DPO Training → Nemotron Evaluation

### Prerequisites

```bash
# Setup environment
conda activate cs224r-project

# Set environment variables
export NVIDIA_API_KEY="your_nvidia_api_key_here"
```

### DPO Training

Train a DPO model starting from an SFT (Supervised Fine-Tuned) model:

```bash
# Basic DPO training with recommended settings
./dpo_qwen.sh

# Or run manually with custom parameters:
python src/main_dpo.py \
    --model_path "anatal/qwen2_05_smol-smoltalk" \
    --dataset_name "HuggingFaceH4/ultrafeedback_binarized" \
    --output_dir "./qwen2_dpo" \
    --epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --lr 5e-7 \
    --warmup_ratio 0.1 \
    --max_length 512 \
    --beta 0.3 \
    --subset 10000 \
    --log_every 50 \
    --save_every 1000
```

**Training will create directories like:**
- `qwen2_dpo/arcane-comet-2_2025-06-02-1415-PST/step_1000/`
- `qwen2_dpo/arcane-comet-2_2025-06-02-1415-PST/final/`

### Start VLLM Server for Inference
After training is done we do inference with VLLM.

```bash
# Start VLLM server with your trained DPO model
./start_vllm.sh ./qwen2_dpo/arcane-comet-2_2025-06-02-1415-PST/final

# Or with custom port:
vllm serve ./qwen2_dpo/arcane-comet-2_2025-06-02-1415-PST/final \
    --host 0.0.0.0 \
    --port 8002 \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len 2048
```

### 3. Run Nemotron Evaluation

```bash
# Quick evaluation with 25 samples
./run_evaluation.sh

# Or with custom sample count:
./run_evaluation.sh 100

# Or run directly:
python evaluation/evaluate_nemotron.py \
    --num_samples 100
```

## Setup Instructions

### Conda Setup

1. **Create and activate a conda environment**:
   ```bash
   # Make the setup script executable
   chmod +x setupenv.sh
   
   # Run the setup script
   ./setupenv.sh
   
   # After the script completes, activate the environment
   conda activate cs224r-project
   ```

### Additional Setup

1.  **Hugging Face Login** 
    ```bash
    huggingface-cli login
    ```
2.  **NVIDIA API Key for Nemotron Reward Model** (for UltraFeedback evaluation):
    - Obtain an API key from [NVIDIA's Llama-3.1-Nemotron-70B-Reward page](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Reward).
    - Set it as an environment variable or manage it securely:
      ```bash
      export NVIDIA_API_KEY="YOUR_API_KEY"
      ```

## Running Experiments

The main scripts for training and evaluation (e.g., `main_sft.py`, `main_dpo.py`) will be located in the `src/` directory.
These scripts will typically be launched from the command line, using arguments to specify configurations, often by pointing to a YAML file in a `src/configs/` directory.

For example, to run an SFT experiment:
```bash
bash src/sft_qwen.sh
```

## Project Structure

```
cs224r-llm-rl-finetuning/
├── src/                     # Core algorithms
│   ├── methods/dpo.py       # DPO implementation
│   ├── data_handling/       # Dataset processing
│   ├── utils/              # Utilities
│   └── main_dpo.py         # Main DPO training script
├── evaluation/             # Evaluation code
│   └── evaluate_nemotron.py # Nemotron evaluation
├── inference/              # Inference utilities
│   └── run_inference.py    # VLLM inference helpers
├── qwen2_dpo/              # Training outputs
│   └── arcane-comet-2_2025-06-02-1415-PST/  # Run-specific directories
│       ├── step_1000/      # Checkpoint saves
│       ├── step_2000/      
│       └── final/          # Final model
├── dpo_qwen.sh            # DPO training script
├── start_vllm.sh          # VLLM server startup
├── run_evaluation.sh      # Evaluation script
├── kill_vllm.sh           # VLLM cleanup script
└── README.md
```
