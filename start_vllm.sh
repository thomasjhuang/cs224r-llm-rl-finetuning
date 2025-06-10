#!/bin/bash

# Simple VLLM startup script with readiness check
# Usage: ./start_vllm.sh <model_path_or_hf_id> [port] [gpu_ids_or_tensor_parallel_size]
# This script starts an OpenAI-compatible server.

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_path_or_hf_id> [port] [tensor_parallel_size]"
    echo "Example (local path, 1 GPU): $0 ./qwen2_dpo/run_20250602_19165"
    echo "Example (HF ID, 8 GPUs): $0 thomasjhuang/qwen2-rloo-countdown 8000 8"
    exit 1
fi

MODEL_PATH="$1"
PORT="${2:-8000}"
TP_SIZE="${3:-1}" # Default to 1 GPU (no tensor parallelism)

echo "🚀 Starting VLLM OpenAI-compatible server..."
echo "   Model: $MODEL_PATH"
echo "   Port: $PORT"
if [ "$TP_SIZE" -gt 1 ]; then
    echo "   Mode: Tensor Parallel"
    echo "   GPUs: $TP_SIZE"
else
    echo "   Mode: Single GPU"
    echo "   GPU: 0 (default)"
fi
echo "==============================================="

# Construct the command
CMD="python -m vllm.entrypoints.openai.api_server --model \"$MODEL_PATH\" --dtype=half --port $PORT --host 0.0.0.0"

if [ "$TP_SIZE" -gt 1 ]; then
    # For tensor parallelism, vLLM manages GPUs. Don't set CUDA_VISIBLE_DEVICES.
    CMD+=" --tensor-parallel-size $TP_SIZE"
else
    # For single GPU, explicitly set the device.
    CMD="CUDA_VISIBLE_DEVICES=0 $CMD"
fi

# Start VLLM in background, redirecting all output to a log file
LOG_FILE="vllm_startup.log"
echo "Redirecting startup logs to: $LOG_FILE"
eval "$CMD" > "$LOG_FILE" 2>&1 &
VLLM_PID=$!

echo "VLLM Command: $CMD"
echo "VLLM PID: $VLLM_PID"
echo "⏳ Waiting for VLLM to be ready..."

# Wait for VLLM to be ready (up to 5 minutes)
for i in {1..60}; do
    sleep 5
    if curl -s -f http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "✅ VLLM is ready and responding on port $PORT"
        echo "🔗 API endpoint: http://localhost:$PORT/v1/chat/completions"
        echo "📊 Health check: http://localhost:$PORT/health"
        echo "==============================================="
        echo "VLLM is running! Use Ctrl+C to stop."
        wait $VLLM_PID
        exit 0
    fi
    echo "   Still starting... ($((i*5))s elapsed)"
done

echo "❌ VLLM failed to start after 5 minutes"
kill $VLLM_PID 2>/dev/null
exit 1 