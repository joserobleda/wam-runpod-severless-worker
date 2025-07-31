#!/bin/bash

# CogVideoX-5b RunPod Worker Startup Script
set -e

echo "🚀 Starting CogVideoX-5b RunPod Worker..."

# Basic environment
export WORKER_DIR=${WORKER_DIR:-"/app"}

# Basic CUDA optimization
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export CUDA_VISIBLE_DEVICES="0"

echo "✅ Environment configured for CogVideoX"

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
python3 -c "import torch; print(f'PyTorch: {torch.__version__} (CUDA: {torch.version.cuda})')" || echo "❌ PyTorch import failed"
python3 -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')" || echo "❌ Diffusers import failed"
python3 -c "import runpod; print(f'RunPod: {runpod.__version__}')" || echo "❌ RunPod import failed"
python3 -c "import xformers; print(f'xFormers: {xformers.__version__}')" || echo "⚠️ xFormers not available (will use standard attention)"

# Quick diffusers test (don't actually load the model, just test the import)
echo "🔧 Testing CogVideoX components..."
python3 -c "from diffusers import CogVideoXPipeline; print('✅ CogVideoXPipeline import successful')" || echo "❌ CogVideoXPipeline import failed"
python3 -c "from diffusers import AutoencoderKLCogVideoX; print('✅ AutoencoderKLCogVideoX import successful')" || echo "❌ AutoencoderKLCogVideoX import failed"
python3 -c "from diffusers import CogVideoXTransformer3DModel; print('✅ CogVideoXTransformer3DModel import successful')" || echo "❌ CogVideoXTransformer3DModel import failed"
python3 -c "from transformers import T5EncoderModel, T5Tokenizer; print('✅ T5 components import successful')" || echo "❌ T5 components import failed"

# Download the model if it doesn't exist
echo "📥 Checking model cache..."
cd "${WORKER_DIR}"

if [ ! -d "model_cache" ] || [ -z "$(ls -A model_cache 2>/dev/null)" ]; then
    echo "📥 Model not found, downloading CogVideoX-5b..."
    python3 download_model.py
    if [ $? -ne 0 ]; then
        echo "❌ Model download failed!"
        exit 1
    fi
else
    echo "✅ Model cache found, skipping download"
fi

# Verify model is working
echo "🔍 Quick model verification..."
python3 -c "
import os
os.chdir('${WORKER_DIR}')
from predict import Predictor
print('✅ Model verification passed')
" || {
    echo "❌ Model verification failed, re-downloading..."
    rm -rf model_cache
    python3 download_model.py
    if [ $? -ne 0 ]; then
        echo "❌ Model re-download failed!"
        exit 1
    fi
}

# Start the handler
echo "🎬 Starting CogVideoX handler..."
exec python3 -u rp_handler.py 