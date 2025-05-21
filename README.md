# CS224R RL Fine-Tuning Project

This project focuses on implementing and exploring Reinforcement Learning algorithms for fine-tuning Large Language Models (LLMs), specifically Qwen 2.5 0.5B Base.
## Project Goal

The primary goal of the initial implementation is to:
1.  Set up data loading pipelines for Instruction Following (UltraFeedback, SmolTalk) and Math Reasoning (Countdown, WarmStart, TinyZero prompts) tasks.
2.  Implement Supervised Fine-Tuning (SFT).
3.  Implement Direct Preference Optimization (DPO).
4.  Implement REINFORCE Leave One-Out (RLOO) with Bradley-Terry reward modeling for preference data and a rule-based reward for math tasks.
5.  Establish an evaluation setup for both tasks using the specified metrics and models.

## Setup Instructions

1.  **Create a Virtual Environment** (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate 
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Hugging Face Login** (if needed for specific models/datasets, though Qwen 2.5 0.5B is public):
    ```bash
    huggingface-cli login
    ```
4.  **NVIDIA API Key for Nemotron Reward Model** (for UltraFeedback evaluation):
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
python src/main_sft.py --config_file src/configs/sft_instruction_following.yaml
```
