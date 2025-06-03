#!/bin/bash

# Nemotron evaluation script (assumes VLLM is already running)
# Usage: ./run_evaluation.sh [num_samples] [port] [model_path] [model_type]

NUM_SAMPLES="${1:-25}"
PORT="${2:-8002}"
MODEL_PATH="${3:-}" # If empty, attempts to auto-detect DPO model
MODEL_TYPE="${4:-DPO}" # Default to DPO if not specified

# Auto-detect latest DPO model if MODEL_PATH is not provided and MODEL_TYPE is DPO
if [ -z "$MODEL_PATH" ] && [ "$MODEL_TYPE" == "DPO" ]; then
    if [ -d "qwen2_dpo" ]; then
        # Find the most recent directory in qwen2_dpo
        LATEST_DIR=$(ls -t qwen2_dpo/ | head -1)
        if [ ! -z "$LATEST_DIR" ]; then
            # Check if it has a 'final' subdirectory (new format) or use directly (old format)
            if [ -d "qwen2_dpo/$LATEST_DIR/final" ]; then
                MODEL_PATH="qwen2_dpo/$LATEST_DIR/final"
            else
                MODEL_PATH="qwen2_dpo/$LATEST_DIR"
            fi
            echo "🔍 Auto-detected DPO model: $MODEL_PATH"
        else
            echo "❌ No DPO model found in qwen2_dpo/ for auto-detection."
            echo "   Please provide a MODEL_PATH if not evaluating a DPO model or if auto-detection fails."
            exit 1
        fi
    else
        echo "❌ qwen2_dpo directory not found for DPO auto-detection."
        echo "   Please provide a MODEL_PATH."
        exit 1
    fi
elif [ -z "$MODEL_PATH" ]; then
    echo "❌ MODEL_PATH (argument 3) is required if MODEL_TYPE is not DPO or for specific model evaluation."
    exit 1
fi

echo "🔍 Running Nemotron Evaluation for $MODEL_TYPE model"
echo "   Samples: $NUM_SAMPLES"
echo "   VLLM Port: $PORT"
echo "   Model Path: $MODEL_PATH"
echo "==============================================="

# Quick check if VLLM is running
echo "🔌 Checking VLLM connection..."
if curl -s -f http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "✅ VLLM is responding on port $PORT"
else
    echo "❌ VLLM not responding on port $PORT"
    echo "   Make sure VLLM is running first with the correct model:"
    echo "   E.g., ./start_vllm.sh $MODEL_PATH"
    exit 1
fi

echo "📊 Starting evaluation..."
echo "==============================================="

python evaluation/evaluate_nemotron.py \
    --model_path_to_evaluate "$MODEL_PATH" \
    --num_samples $NUM_SAMPLES \
    --vllm_port $PORT \
    --model_type "$MODEL_TYPE"

echo "==============================================="
echo "✅ Evaluation complete!" 