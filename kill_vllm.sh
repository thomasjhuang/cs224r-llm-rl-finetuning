#!/bin/bash

echo "🔥 Killing VLLM and cleaning GPU processes..."
echo "=============================================="

# Kill VLLM processes by name
echo "🎯 Killing VLLM processes..."
pkill -f "vllm" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ VLLM processes killed"
else
    echo "ℹ️  No VLLM processes found"
fi

# Kill Python processes that might be using GPU for inference
echo "🐍 Killing Python GPU processes..."
pkill -f "python.*serve" 2>/dev/null
pkill -f "python.*inference" 2>/dev/null

# Get PIDs of processes using GPU and kill them
echo "🔧 Cleaning GPU processes..."
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr '\n' ' ')

if [ ! -z "$GPU_PIDS" ]; then
    echo "📋 Found GPU processes: $GPU_PIDS"
    for pid in $GPU_PIDS; do
        if [ ! -z "$pid" ]; then
            echo "   Killing PID: $pid"
            kill -9 $pid 2>/dev/null
        fi
    done
    echo "✅ GPU processes cleaned"
else
    echo "ℹ️  No GPU processes found"
fi

# Wait a moment for cleanup
sleep 2

# Check final status
echo "=============================================="
echo "🔍 Final GPU status:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || echo "No GPU processes running"
echo "=============================================="
echo "✅ Cleanup complete!" 