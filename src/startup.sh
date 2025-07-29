#!/bin/bash

# Wan 2.2 TI2V-5B RunPod Serverless Worker Startup
# Optimized for reduced image size and faster cold starts

set -e

echo "🚀 Starting Wan 2.2 TI2V-5B RunPod Worker..."
echo "================================"

# Environment variables
export WORKER_DIR=${WORKER_DIR:-"/app"}
export WORKER_MODEL_DIR=${WORKER_MODEL_DIR:-"/app/model"}
export WORKER_USE_CUDA=${WORKER_USE_CUDA:-"True"}

# GPU information
echo "📊 System Information:"
echo "  Python: $(python3 --version)"
echo "  Working Directory: $(pwd)"
echo "  User: $(whoami)"

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
    echo "  VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MB"
    echo "  Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1)"
else
    echo "  GPU: CUDA not available"
fi

# Directory setup
echo ""
echo "📁 Setting up directories..."

# Create necessary directories with proper permissions
mkdir -p "${WORKER_DIR}/logs"
mkdir -p "/tmp/video_generation"
mkdir -p "/home/user/.cache"

# Set permissions
chmod 755 "${WORKER_DIR}"
chmod 755 "/tmp/video_generation"

echo "✅ Directories created successfully"

# Model verification
echo ""
echo "🤖 Verifying model setup..."

if [ ! -d "${WORKER_MODEL_DIR}/wan2.2-ti2v-5b" ]; then
    echo "❌ Model directory not found: ${WORKER_MODEL_DIR}/wan2.2-ti2v-5b"
    echo "Please ensure the Wan 2.2 model was downloaded during Docker build"
    exit 1
fi

echo "✅ Model directory found: ${WORKER_MODEL_DIR}/wan2.2-ti2v-5b"

# Check for key model files
KEY_FILES=("config.json")
for file in "${KEY_FILES[@]}"; do
    if [ -f "${WORKER_MODEL_DIR}/wan2.2-ti2v-5b/${file}" ]; then
        echo "  ✅ ${file}"
    else
        echo "  ⚠️  ${file} (missing but might be optional)"
    fi
done

# Wan 2.2 code verification
echo ""
echo "🔧 Verifying Wan 2.2 code..."

if [ ! -f "${WORKER_DIR}/wan2.2_code/generate.py" ]; then
    echo "❌ Wan 2.2 generate.py not found: ${WORKER_DIR}/wan2.2_code/generate.py"
    exit 1
fi

echo "✅ Wan 2.2 code found: ${WORKER_DIR}/wan2.2_code/generate.py"

# Set Python path to include the Wan 2.2 code directory
export PYTHONPATH="${WORKER_DIR}/wan2.2_code:${PYTHONPATH}"
echo "✅ Python path configured: ${PYTHONPATH}"

# Memory optimization
echo ""
echo "🧠 Memory optimization..."

# Set memory-efficient environment variables
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export CUDA_VISIBLE_DEVICES="0"
export OMP_NUM_THREADS="4"

# Python optimization flags
export PYTHONOPTIMIZE="1"
export PYTHONDONTWRITEBYTECODE="1"

echo "✅ Memory optimization configured"

# Python dependencies check
echo ""
echo "🐍 Checking Python dependencies..."

python3 -c "
import sys
try:
    import torch
    print(f'✅ PyTorch {torch.__version__}')
    if torch.cuda.is_available():
        print(f'✅ CUDA {torch.version.cuda}')
        print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('⚠️  CUDA not available')
    
    import transformers
    print(f'✅ Transformers {transformers.__version__}')
    
    import diffusers
    print(f'✅ Diffusers {diffusers.__version__}')
    
    import runpod
    print(f'✅ RunPod {runpod.__version__}')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
" || exit 1

# Runtime optimizations
echo ""
echo "⚡ Applying runtime optimizations..."

# Pre-allocate some GPU memory to avoid fragmentation
if [ "$WORKER_USE_CUDA" = "True" ]; then
    python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # Pre-warm CUDA
    x = torch.randn(1, device='cuda')
    print('✅ CUDA pre-warmed')
    del x
    torch.cuda.empty_cache()
else:
    print('⚠️  CUDA not available for pre-warming')
" || echo "⚠️  CUDA pre-warming failed"
fi

# File permissions final check
echo ""
echo "🔐 Final permissions check..."
chmod +x "${WORKER_DIR}/rp_handler.py" 2>/dev/null || true
chmod +x "${WORKER_DIR}/predict.py" 2>/dev/null || true

echo "✅ Permissions set"

# Start the handler
echo ""
echo "🎬 Starting Wan 2.2 TI2V-5B RunPod Handler..."
echo "================================"

cd "${WORKER_DIR}"

# Run with optimized settings
exec python3 -u rp_handler.py 