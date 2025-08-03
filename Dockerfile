# Use the official RunPod base image with CUDA support
FROM runpod/base:0.6.2-cuda12.2.0

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HUB_CACHE=/runpod-volume/.cache/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libfontconfig1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA support first (using correct index URL with compatible versions)
RUN pip install --no-cache-dir --verbose --root-user-action=ignore \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121 && \
    echo "Successfully installed PyTorch with CUDA support" && \
    pip list | grep torch

# Copy and install remaining Python dependencies
COPY builder/requirements.txt /tmp/requirements.txt

# Create a modified requirements file without PyTorch packages
RUN grep -v -E "^torch|^torchvision|^torchaudio" /tmp/requirements.txt > /tmp/requirements_no_torch.txt

# Install remaining dependencies
RUN pip install --no-cache-dir --verbose --root-user-action=ignore -r /tmp/requirements_no_torch.txt && \
    echo "Successfully installed remaining requirements" && \
    pip list | grep -E "(runpod|diffusers)"

# Install additional optimizations for video processing
RUN pip install --no-cache-dir --verbose --root-user-action=ignore \
    av \
    decord && \
    echo "Successfully installed additional video processing packages"

# Verify critical packages are installed
RUN echo "=== Verifying Package Installations ===" && \
    python3 -c "import torch; print(f'✓ PyTorch version: {torch.__version__}'); print(f'✓ CUDA available: {torch.cuda.is_available()}')" && \
    python3 -c "import runpod; print(f'✓ RunPod version: {runpod.__version__}')" && \
    python3 -c "import diffusers; print(f'✓ Diffusers version: {diffusers.__version__}')" && \
    echo "✓ All critical packages verified successfully"

# Create necessary directories
RUN mkdir -p /builder /src /runpod-volume/model /runpod-volume/outputs /runpod-volume/.cache

# Copy builder scripts
COPY builder/ /builder/
RUN chmod +x /builder/download_model.py

# Copy source code
COPY src/ /src/
RUN chmod +x /src/handler.py

# Set working directory
WORKDIR /src

# Create a health check script
RUN echo '#!/bin/bash\npython3 -c "from src.handler import health_check; import json; print(json.dumps(health_check(), indent=2))"' > /health_check.sh
RUN chmod +x /health_check.sh

# Set up the entrypoint
COPY <<EOF /entrypoint.sh
#!/bin/bash
set -e

echo "Starting Wan2.2 RunPod Serverless Worker..."
echo "Python version: \$(python3 --version)"
echo "PyTorch version: \$(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA available: \$(python3 -c 'import torch; print(torch.cuda.is_available())')"

# Set up Python path
export PYTHONPATH="/src:/builder:\$PYTHONPATH"

# Start the handler
cd /src
exec python3 handler.py
EOF

RUN chmod +x /entrypoint.sh

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD /health_check.sh || exit 1

# Expose the port (RunPod uses dynamic ports, but this is for documentation)
EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Labels for better organization
LABEL maintainer="RunPod Deployment"
LABEL description="Wan-AI/Wan2.2-T2V-A14B-Diffusers text-to-video generation"
LABEL version="1.0.0"
LABEL model="Wan-AI/Wan2.2-T2V-A14B-Diffusers" 