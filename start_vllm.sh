#!/bin/bash

# Simple VLLM startup script with readiness check
# Usage: ./start_vllm.sh <model_path> [port] [gpu_id]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_path> [port] [gpu_id]"
    echo "Example: $0 ./qwen2_dpo/run_20250602_19165"
    echo "Example: $0 ./qwen2_dpo/run_20250602_19165 8002 0"
    exit 1
fi

MODEL_PATH="$1"
PORT="${2:-8002}"
GPU_ID="${3:-0}"

echo "🚀 Starting VLLM server..."
echo "   Model: $MODEL_PATH"
echo "   Port: $PORT"
echo "   GPU: $GPU_ID"
echo "==============================================="

# Start VLLM in background
CUDA_VISIBLE_DEVICES=$GPU_ID vllm serve "$MODEL_PATH" --dtype=half --port $PORT &
VLLM_PID=$!

echo "VLLM PID: $VLLM_PID"
echo "⏳ Waiting for VLLM to be ready..."

# Wait for VLLM to be ready (up to 2 minutes)
for i in {1..24}; do
    sleep 5
    if curl -s -f http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "✅ VLLM is ready and responding on port $PORT"
        echo "🔗 API endpoint: http://localhost:$PORT"
        echo "📊 Health check: http://localhost:$PORT/health"
        echo "==============================================="
        echo "VLLM is running! Use Ctrl+C to stop."
        wait $VLLM_PID
        exit 0
    fi
    echo "   Still starting... ($((i*5))s elapsed)"
done

echo "❌ VLLM failed to start after 2 minutes"
kill $VLLM_PID 2>/dev/null
exit 1 