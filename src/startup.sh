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
    echo "🔧 GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MB"
    echo "🔧 GPU Utilization: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)%"
else
    echo "⚠️  No GPU detected - will use CPU"
fi

# Quick Python/package check
echo "🔧 Python version: $(python3 --version)"
echo "🔧 Checking key packages..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" || echo "❌ PyTorch import failed"
python3 -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')" || echo "❌ Diffusers import failed"
python3 -c "import runpod; print(f'RunPod: {runpod.__version__}')" || echo "❌ RunPod import failed"

# Quick diffusers test (don't actually load the model, just test the import)
echo "🔧 Testing diffusers Wan2.2 components..."
python3 -c "from diffusers import WanPipeline; print('✅ WanPipeline import successful')" || echo "❌ WanPipeline import failed"
python3 -c "from diffusers import AutoencoderKLWan; print('✅ AutoencoderKLWan import successful')" || echo "❌ AutoencoderKLWan import failed"
python3 -c "from diffusers import WanTransformer3DModel; print('✅ WanTransformer3DModel import successful')" || echo "❌ WanTransformer3DModel import failed"
python3 -c "from diffusers import UniPCMultistepScheduler; print('✅ UniPCMultistepScheduler import successful')" || echo "❌ UniPCMultistepScheduler import failed"

# Start the handler
echo "🎬 Starting Diffusers-based handler..."
cd "${WORKER_DIR}"
exec python3 -u rp_handler.py 