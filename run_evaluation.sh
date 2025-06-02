#!/bin/bash

# Nemotron evaluation script (assumes VLLM is already running)
# Usage: ./run_evaluation.sh [num_samples] [port]

NUM_SAMPLES="${1:-25}"
PORT="${2:-8002}"

echo "🔍 Running Nemotron Evaluation"
echo "   Samples: $NUM_SAMPLES"
echo "   VLLM Port: $PORT"
echo "==============================================="

# Quick check if VLLM is running
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
    --run_ultrafeedback_eval \
    --num_samples_ultrafeedback_winrate $NUM_SAMPLES

echo "==============================================="
echo "✅ Evaluation complete!" 