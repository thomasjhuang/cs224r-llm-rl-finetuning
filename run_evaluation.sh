#!/bin/bash

# Nemotron evaluation script (assumes VLLM is already running)
# Usage: ./run_evaluation.sh [mode] [num_samples] [port] [model_path] [model_type]
# If model_path is omitted or set to "auto", it will be auto-detected from the running VLLM server

MODE="${1:?Mode argument is required (collect_dpo, collect_ref, compare)}"
NUM_SAMPLES="${2:-25}"
PORT="${3:-8002}" # Port for the model being evaluated (DPO or SFT)
MODEL_PATH="${4:-auto}" # If empty or "auto", attempts to auto-detect from VLLM
MODEL_TYPE="${5:-auto}" # If empty or "auto", attempts to auto-detect

# Handle auto-detection case
if [ "$MODEL_PATH" == "auto" ] || [ -z "$MODEL_PATH" ]; then
    echo "🔍 Model path set to auto-detect from running VLLM server"
    MODEL_PATH_ARG=""
else
    MODEL_PATH_ARG="--model_path_to_evaluate $MODEL_PATH"
fi

if [ "$MODEL_TYPE" == "auto" ] || [ -z "$MODEL_TYPE" ]; then
    echo "🔍 Model type set to auto-detect from running VLLM server"
    MODEL_TYPE_ARG=""
else
    MODEL_TYPE_ARG="--model_type $MODEL_TYPE"
fi

# Auto-detect latest DPO model if MODEL_PATH is not provided and we're not auto-detecting
if [ "$MODEL_PATH" != "auto" ] && [ -z "$MODEL_PATH" ] && [ "$MODEL_TYPE" == "DPO" ]; then
    if [ -d "qwen2_dpo" ]; then
        # Find the most recent run directory in qwen2_dpo
        LATEST_RUN_DIR=$(ls -t qwen2_dpo/ | head -1)
        if [ ! -z "$LATEST_RUN_DIR" ]; then
            # Find the most recent model directory within the run directory (e.g., a timestamped or named sub-run)
            # This handles structures like qwen2_dpo/run_XXXX/sub_run_YYYY/final
            LATEST_MODEL_PARENT_DIR=$(ls -t "qwen2_dpo/$LATEST_RUN_DIR/" | head -1)
            
            # Check if it has a 'final' subdirectory (new format)
            if [ -d "qwen2_dpo/$LATEST_RUN_DIR/$LATEST_MODEL_PARENT_DIR/final" ]; then
                MODEL_PATH="qwen2_dpo/$LATEST_RUN_DIR/$LATEST_MODEL_PARENT_DIR/final"
            # Check if LATEST_MODEL_PARENT_DIR itself is 'final' (older format, direct run_XXXX/final)
            elif [ "$LATEST_MODEL_PARENT_DIR" == "final" ] && [ -d "qwen2_dpo/$LATEST_RUN_DIR/final" ]; then
                 MODEL_PATH="qwen2_dpo/$LATEST_RUN_DIR/final"
            # Check if LATEST_MODEL_PARENT_DIR is a completed run and contains 'final' (e.g. brisk-lion-32.../final)
            elif [ -d "qwen2_dpo/$LATEST_RUN_DIR/$LATEST_MODEL_PARENT_DIR" ] && [ "$(basename "$LATEST_MODEL_PARENT_DIR")" != "final" ] && [ -d "qwen2_dpo/$LATEST_RUN_DIR/$LATEST_MODEL_PARENT_DIR/final" ]; then
                 MODEL_PATH="qwen2_dpo/$LATEST_RUN_DIR/$LATEST_MODEL_PARENT_DIR/final"
            # Fallback to LATEST_MODEL_PARENT_DIR if it's a directory (older format with no 'final' subfolder, e.g. step_XXX)
            # Or if LATEST_RUN_DIR itself is the model dir (very old format)
            elif [ -d "qwen2_dpo/$LATEST_RUN_DIR/$LATEST_MODEL_PARENT_DIR" ]; then
                MODEL_PATH="qwen2_dpo/$LATEST_RUN_DIR/$LATEST_MODEL_PARENT_DIR"
            elif [ -d "qwen2_dpo/$LATEST_RUN_DIR" ]; then # Handles case where LATEST_DIR is the model dir itself.
                MODEL_PATH="qwen2_dpo/$LATEST_RUN_DIR"
            fi
            
            if [ ! -z "$MODEL_PATH" ] && [ -d "$MODEL_PATH" ] ; then
                echo "🔍 Auto-detected DPO model from filesystem: $MODEL_PATH"
                MODEL_PATH_ARG="--model_path_to_evaluate $MODEL_PATH"
            else
                echo "❌ Could not reliably auto-detect DPO model structure in qwen2_dpo/$LATEST_RUN_DIR/$LATEST_MODEL_PARENT_DIR."
                MODEL_PATH="" # Reset on failure
            fi
        fi
        
        if [ -z "$MODEL_PATH" ]; then
            echo "❌ No DPO model found in qwen2_dpo/ for auto-detection or detection failed."
            echo "   Will attempt to auto-detect from running VLLM server."
        fi
    else
        echo "❌ qwen2_dpo directory not found for DPO auto-detection."
        echo "   Will attempt to auto-detect from running VLLM server."
    fi
fi

echo "🔍 Running Nemotron Evaluation"
echo "   Mode: $MODE"
echo "   Samples: $NUM_SAMPLES"
echo "   VLLM Port (for evaluated model): $PORT"
if [ ! -z "$MODEL_PATH_ARG" ]; then
    echo "   Model Path: $MODEL_PATH"
else
    echo "   Model Path: [Auto-detect from VLLM]"
fi
if [ ! -z "$MODEL_TYPE_ARG" ]; then
    echo "   Model Type: $MODEL_TYPE"
else
    echo "   Model Type: [Auto-detect from VLLM]"
fi
echo "==============================================="

if [ "$MODE" == "collect_dpo" ] || [ "$MODE" == "collect_ref" ]; then
    echo "🔌 Checking VLLM connection on port $PORT..."
    if curl -s -f http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "✅ VLLM is responding on port $PORT"
    else
        echo "❌ VLLM not responding on port $PORT"
        echo "   Make sure VLLM is running first with the correct model."
        if [ ! -z "$MODEL_PATH_ARG" ]; then
            echo "   E.g., ./start_vllm.sh $MODEL_PATH $PORT"
        else
            echo "   E.g., ./start_vllm.sh <your_model_path> $PORT"
        fi
        exit 1
    fi
fi

echo "📊 Starting evaluation (Mode: $MODE)..."
echo "==============================================="

# Build the command with optional arguments
CMD="python evaluation/evaluate_nemotron.py --mode $MODE --num_samples $NUM_SAMPLES --vllm_port $PORT"

if [ ! -z "$MODEL_PATH_ARG" ]; then
    CMD="$CMD $MODEL_PATH_ARG"
fi

if [ ! -z "$MODEL_TYPE_ARG" ]; then
    CMD="$CMD $MODEL_TYPE_ARG"
fi

# Note: --reference_vllm_port will be used by evaluate_nemotron.py if mode is collect_ref or compare
# It defaults to 8003 in the python script.
eval $CMD

echo "==============================================="
echo "✅ Evaluation phase ($MODE) complete!" 