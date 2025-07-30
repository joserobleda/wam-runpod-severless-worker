#!/bin/bash

# Ultra-simple Wan 2.2 TI2V-5B RunPod Worker (Diffusers)
set -e

echo "🚀 Starting Wan 2.2 TI2V-5B RunPod Worker (Diffusers approach)..."

# Basic environment
export WORKER_DIR=${WORKER_DIR:-"/app"}

# Basic CUDA optimization
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export CUDA_VISIBLE_DEVICES="0"

echo "✅ Environment configured for Diffusers"

# Quick GPU check
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "🔧 GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
else
    echo "⚠️  No GPU detected - will use CPU"
fi

# Start the handler
echo "🎬 Starting Diffusers-based handler..."
cd "${WORKER_DIR}"
exec python3 -u rp_handler.py 