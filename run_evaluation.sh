#!/bin/bash

# Nemotron evaluation script (assumes VLLM is already running)
# Usage: ./run_evaluation.sh [num_samples] [port] [model_path]

NUM_SAMPLES="${1:-25}"
PORT="${2:-8002}"
MODEL_PATH="${3:-}"

# Auto-detect latest DPO model if not provided
if [ -z "$MODEL_PATH" ]; then
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
            echo "❌ No DPO model found in qwen2_dpo/"
            exit 1
        fi
    else
        echo "❌ qwen2_dpo directory not found"
        exit 1
    fi
fi

echo "🔍 Running Nemotron Evaluation"
echo "   Samples: $NUM_SAMPLES"
echo "   VLLM Port: $PORT"
echo "   DPO Model: $MODEL_PATH"
echo "==============================================="

echo "🔌 Checking VLLM connection..."
if curl -s -f http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "✅ VLLM is responding on port $PORT"
else
    echo "❌ VLLM not responding on port $PORT"
    echo "   Make sure VLLM is running first:"
    echo "   ./start_vllm.sh <model_path>"
    exit 1
fi

echo "📊 Starting evaluation..."
echo "==============================================="

python evaluation/evaluate_nemotron.py \
    --dpo_model "$MODEL_PATH" \
    --num_samples $NUM_SAMPLES \
    --vllm_port $PORT

echo "==============================================="
echo "✅ Evaluation complete!" 