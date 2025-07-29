#!/bin/bash

# Exit on any error
set -e

echo "🚀 Starting Wan 2.2 TI2V-5B RunPod Serverless Worker..."

# Ensure necessary directories exist
mkdir -p /home/user/.triton/autotune
mkdir -p /home/user/.cache/matplotlib
mkdir -p /home/user/.cache/huggingface
mkdir -p /home/user/.cache/torch
mkdir -p /tmp/video_generation

# Set proper permissions
chown -R user:user /home/user/.triton /home/user/.cache /tmp/video_generation 2>/dev/null || true

# Set environment variables
export MPLBACKEND=Agg
export TRITON_CACHE_DIR=/home/user/.triton
export HF_HOME=/home/user/.cache/huggingface
export TORCH_HOME=/home/user/.cache/torch
export TMPDIR=/tmp/video_generation

# GPU optimization settings
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Disable some warnings for cleaner logs
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

# Check if model directory exists
if [ ! -d "${WORKER_MODEL_DIR}/wan2.2-ti2v-5b" ]; then
    echo "❌ Model directory not found: ${WORKER_MODEL_DIR}/wan2.2-ti2v-5b"
    echo "Please ensure the Wan 2.2 model was downloaded during Docker build"
    exit 1
fi

echo "✅ Model directory found: ${WORKER_MODEL_DIR}/wan2.2-ti2v-5b"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "🔧 GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    echo ""
else
    echo "⚠️ nvidia-smi not available, running without GPU info"
fi

# Check available disk space
echo "💾 Disk Space:"
df -h /tmp
echo ""

# Set Python path to include the Wan 2.2 model directory
export PYTHONPATH="${WORKER_MODEL_DIR}/wan2.2-ti2v-5b:${PYTHONPATH}"

# Change to the working directory
cd /app

echo "🎬 Starting video generation service..."
echo "Model directory: ${WORKER_MODEL_DIR}"
echo "Worker directory: /app"
echo "Python path: ${PYTHONPATH}"
echo ""

# Start the main application with error handling
exec python3 -u /app/rp_handler.py --model-dir="${WORKER_MODEL_DIR}" 2>&1 